import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import ephem
from math import degrees
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Add to imports at top

# Function to fetch Daymet data for specific days within a year

def fetch_daymet_by_days(lat, lon, year, days):

    """
    Fetch Daymet data for a specific latitude, longitude, year, and specific days.
    """
    url = "https://daymet.ornl.gov/single-pixel/api/data"
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    params = {
        "lat": lat,
        "lon": lon,
        "vars": "T2MWET,QV2M,RH2M,T2M_MAX,ALLSKY_SFC_SW_DWN,PS,T2MDEW,WS2M,T2M_MIN,T2M,PRECTOTCORR",
        "start": start_date,
        "end": end_date,
        "format": "csv"
    }
    response = requests.get(url, params=params)
    print(response)
    if response.status_code == 200:
        daymet_data = pd.read_csv(StringIO(response.text), skiprows=6)  # Adjust skiprows for header lines
        # print(daymet_data[daymet_data['yday'].isin(days)])
        return daymet_data[daymet_data['yday'].isin(days)]
    else:
        print(f"Error fetching data for {lat}, {lon} in year {year}: {response.status_code}")
        return None

def create_daymet_dataframe(weather_data, metadata):
    """
    Merge weather_data with metadata, fetch Daymet data by specific days, and return a new DataFrame.
    """
    # Ensure Env is the index for metadata for easy mapping
    if "Env" not in metadata.index:
        metadata.set_index("Env", inplace=True)

    # Convert Date column to string
    weather_data["Date"] = weather_data["Date"].astype(str)

    # Cache for storing fetched Daymet data by (lat, lon, year)
    daymet_cache = {}

    # Initialize list for storing results
    results = []

    # Group weather_data by environment and year
    for (env, year), group in tqdm(weather_data.groupby(["Env", weather_data["Date"].str[:4].astype(int)])):
        try:
            # Get latitude and longitude for the current environment
            lat = metadata.loc[env, "Weather_Station_Latitude (in decimal numbers NOT DMS)"]
            lon = metadata.loc[env, "Weather_Station_Longitude (in decimal numbers NOT DMS)"]
        except KeyError:
            print(f"Environment '{env}' not found in metadata.")
            continue

        # Extract unique days of the year for the current group
        days = group["Date"].apply(lambda x: datetime.strptime(x, "%Y%m%d").timetuple().tm_yday).unique()

        # Check if data for this (lat, lon, year) is already fetched
        cache_key = (lat, lon, year)
        if cache_key not in daymet_cache:
            daymet_data = fetch_daymet_by_days(lat, lon, year, days)
            if daymet_data is not None:
                daymet_cache[cache_key] = daymet_data
            else:
                continue
        else:
            daymet_data = daymet_cache[cache_key]

        # Add environment and date columns to the filtered Daymet data
        for _, row in group.iterrows():
            date_str = row["Date"]
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            yday = date_obj.timetuple().tm_yday

            # Filter Daymet data for the specific day
            day_data = daymet_data[daymet_data['yday'] == yday]
            if not day_data.empty:
                day_data = day_data.copy()
                day_data["Env"] = env
                day_data["Date"] = date_str
                results.append(day_data)
                # print(results)

    # Combine all results into a single DataFrame
    if results:
        final_df = pd.concat(results, ignore_index=True)
        # Set index to Env and Date
        final_df.set_index(["Env", "Date"], inplace=True)
        return final_df
    else:
        print("No Daymet data matched the input criteria.")
        return pd.DataFrame()
    
def load_senescence_data(file_path):

    plant_data = pd.read_csv(file_path)
    plant_data = plant_data.dropna(subset=['Tos'])
    print(plant_data)

    lat_long = {
        'Toolik': (68.623, -149.606),
        'Coldfoot': (67.25, -150.175),
        'Sagwon': (69.373, -148.700)
    }

    # Add 'lat' and 'long' columns to the DataFrame
    plant_data['lat'] = plant_data['site'].map(lambda site: lat_long[site][0] if site in lat_long else None)
    plant_data['long'] = plant_data['site'].map(lambda site: lat_long[site][1] if site in lat_long else None)
    print(plant_data)
    return plant_data


def compile_weather_data(senescence_df, hourly_weather_path=None):
    """
    Compile weather data for each unique Site, Src, Plot, Ind, Yrm, and Tcode combination.
    Now includes days_until_senescence and doy columns.
    Optionally merges in daily summaries from hourly weather data.
    """
    # Drop unnecessary columns
    senescence_df = senescence_df.drop(columns=["gl"])

    # Cache for storing fetched Daymet data
    daymet_cache = {}
    results = []

    for _, row in tqdm(senescence_df.iterrows(), total=senescence_df.shape[0]):
        site, src, plot, ind, yrm, tcode, tos, lat, lon = (
            row["site"], row["Src"], row["Plot"], row["Ind"], row["Yrm"], row["Tcode"], row["Tos"], row["lat"], row["long"]
        )

        cache_key = (lat, lon, yrm)
        if cache_key not in daymet_cache:
            daymet_data = fetch_daymet_by_days(lat, lon, yrm, days=range(1, 367))
            if daymet_data is not None:
                daymet_cache[cache_key] = daymet_data
            else:
                continue
        else:
            daymet_data = daymet_cache[cache_key]

        # Filter Daymet data for the growing season
        daymet_data = daymet_data[(daymet_data['yday'] >= 150) & (daymet_data['yday'] <= 250)]

        # Add metadata columns and calculate days until senescence
        daymet_data = daymet_data.copy()
        daymet_data["Site"] = site
        daymet_data["Src"] = src
        daymet_data["Plot"] = plot
        daymet_data["Ind"] = ind
        daymet_data["Yrm"] = yrm
        daymet_data["Tcode"] = tcode
        daymet_data["days_until_senescence"] = tos - daymet_data["yday"]
        daymet_data["doy"] = daymet_data["yday"].astype(int)

        results.append(daymet_data)

    if not results:
        print("No weather data compiled.")
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)
    final_df.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", "yday"], inplace=True)

    # --- Merge in daily summaries from hourly weather data if provided ---
    if hourly_weather_path is not None:
        hourly = pd.read_csv(hourly_weather_path)
        # Parse datetime and extract year and doy
        hourly['ts'] = pd.to_datetime(hourly['ts'])
        hourly['year'] = hourly['ts'].dt.year
        hourly['doy'] = hourly['ts'].dt.dayofyear

        # Calculate daily summaries grouped by site, year, doy
        def agg_day(df):
            gdh = df.loc[df['temp'] > 0, 'temp'].sum()  # Growing degree hours (base 0C)
            frost_hours = (df['temp'] < 0).sum()
            snow_amt = df['snow'].sum() if 'snow' in df.columns else pd.NA
            min_sea = df['sea'].min() if 'sea' in df.columns else pd.NA
            max_sea = df['sea'].max() if 'sea' in df.columns else pd.NA
            return pd.Series({
                'growing_degree_hours': gdh,
                'frost_hours': frost_hours,
                'snow_amount': snow_amt,
                'min_sea': min_sea,
                'max_sea': max_sea
            })

        daily = (
            hourly.groupby(['site', 'year', 'doy']).apply(agg_day).reset_index()
        )
        # Map abbreviated site codes to full names
        site_map = {'TL': 'Toolik', 'CF': 'Coldfoot', 'SG': 'Sagwon'}
        daily['site'] = daily['site'].replace(site_map)
        
        # Ensure merge columns are of the same type and format
        final_df = final_df.reset_index()

        # Merge with final_df (index must match)
        merged = final_df.merge(
            daily,
            left_on=['Site', 'Yrm', 'doy'],
            right_on=['site', 'year', 'doy'],
            how='left'
        )
        # Drop duplicate columns from right DataFrame if present
        for col in ['site', 'year', 'year_x', 'year_y']:
            if col in merged.columns:
                merged = merged.drop(columns=col)

        print(33333)
        print(merged)
        print(merged.columns)

        # Restore index
        final_df = merged.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", "yday"])

    return final_df

def sanitize_filename(filename):
    """Convert variable name to a safe filename"""
    # Replace spaces, slashes, and special characters with underscores
    safe_name = re.sub(r'[\s/\\()\[\]{}]+', '_', filename)
    # Remove any other non-alphanumeric characters
    safe_name = re.sub(r'[^\w\-_.]', '', safe_name)
    return safe_name

def plot_environmental_data(weather_df):
    """
    Create plots for each environmental variable with subplots for each site
    """
    plot_dir = "environmental_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    df = weather_df.reset_index()
    
    # Get environmental variables (exclude metadata columns)
    metadata_cols = ['Site', 'Src', 'Plot', 'Ind', 'Yrm', 'Tcode', 'yday', 'year', 'senescence']
    env_vars = [col for col in df.columns if col not in metadata_cols]
    print(f"Plotting variables: {env_vars}")
    
    sites = df['Site'].unique()
    n_sites = len(sites)
    
    # Create one figure per environmental variable with subplots for each site
    for var in env_vars:
        # Calculate subplot grid dimensions
        n_rows = (n_sites + 1) // 2  # Round up division
        n_cols = min(2, n_sites)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(f'{var} across sites', fontsize=16, y=1.02)
        
        # Handle both single row and multiple row cases
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each site in its own subplot
        for idx, site in enumerate(sites):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            site_data = df[df['Site'] == site]
            sns.lineplot(data=site_data, x='yday', y=var, hue='Yrm', ax=ax)
            
            ax.set_title(f'{site}')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel(var)
            
            # Adjust legend
            if len(site_data['Yrm'].unique()) > 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove empty subplots if odd number of sites
        if n_sites % 2 != 0:
            fig.delaxes(axes[-1, -1])
        
        # Save figure with sanitized filename
        safe_var_name = sanitize_filename(var)
        plt.savefig(os.path.join(plot_dir, f'{safe_var_name}_sites_comparison.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

# Main execution
output_file = "data/compiled_weather_data.csv"
hourly_weather_path = "data/AS25_WSdata_GS.csv"

if os.path.exists(output_file):
    print(f"Loading existing weather data from '{output_file}'")
    compiled_weather_data = pd.read_csv(output_file)
    # Restore the index structure
    compiled_weather_data.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", "yday"], inplace=True)
else:
    print("Fetching new weather data...")
    senescence_df = load_senescence_data('data/ToS_per_tiller_2025-05-01.csv')
    compiled_weather_data = compile_weather_data(senescence_df, hourly_weather_path=hourly_weather_path)
    compiled_weather_data.to_csv(output_file)
    print(f"Compiled weather data saved to '{output_file}'")

# Create environmental plots
plot_environmental_data(compiled_weather_data)
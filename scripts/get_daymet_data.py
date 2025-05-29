import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm

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

def compile_weather_data(senescence_df):
    """
    Compile weather data for each unique Site, Src, Plot, Ind, Yrm, and Tcode combination.
    """
    # Drop unnecessary columns
    senescence_df = senescence_df.drop(columns=["gl"])

    # Cache for storing fetched Daymet data by (lat, lon, year)
    daymet_cache = {}

    # Initialize list for storing results
    results = []

    # Iterate over unique combinations of Site, Src, Plot, Ind, Yrm, and Tcode
    for _, row in tqdm(senescence_df.iterrows(), total=senescence_df.shape[0]):
        site, src, plot, ind, yrm, tcode, tos, lat, lon = (
            row["site"], row["Src"], row["Plot"], row["Ind"], row["Yrm"], row["Tcode"], row["Tos"], row["lat"], row["long"]
        )

        # Generate a unique cache key for (lat, lon, year)
        cache_key = (lat, lon, yrm)
        if cache_key not in daymet_cache:
            # Fetch Daymet data for the year
            daymet_data = fetch_daymet_by_days(lat, lon, yrm, days=range(1, 367))  # Fetch all days of the year
            if daymet_data is not None:
                daymet_cache[cache_key] = daymet_data
            else:
                continue
        else:
            daymet_data = daymet_cache[cache_key]

        # Filter Daymet data for the growing season (May 1 to October 31)
        daymet_data = daymet_data[(daymet_data['yday'] >= 121) & (daymet_data['yday'] <= 304)]

        # Add Site, Src, Plot, Ind, Yrm, Tcode, and senescence columns to the Daymet data
        daymet_data = daymet_data.copy()
        daymet_data["Site"] = site
        daymet_data["Src"] = src
        daymet_data["Plot"] = plot
        daymet_data["Ind"] = ind
        daymet_data["Yrm"] = yrm
        daymet_data["Tcode"] = tcode
        daymet_data["senescence"] = daymet_data["yday"] >= tos

        # Append the processed data to results
        results.append(daymet_data)

    # Combine all results into a single DataFrame
    if results:
        final_df = pd.concat(results, ignore_index=True)
        # Set index to Site, Src, Plot, Ind, Yrm, Tcode, and yday for time-series format
        final_df.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", "yday"], inplace=True)
        return final_df
    else:
        print("No weather data compiled.")
        return pd.DataFrame()

# Load senescence data
senescence_df = load_senescence_data('ToS_per_tiller_2025-05-01.csv')

# Compile weather data
compiled_weather_data = compile_weather_data(senescence_df)

# Save the resulting weather data to a CSV file
output_file = "compiled_weather_data.csv"
compiled_weather_data.to_csv(output_file)
print(f"Compiled weather data saved to '{output_file}'")
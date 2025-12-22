import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Determine Time of Senescence (ToS) from phenological data.")
    parser.add_argument("-i", "--input", required=True, help="Input file containing phenological data.")
    parser.add_argument("-o", "--output", required=True, help="Output file to save ToS results.")
    return parser.parse_args()

def fetch_daymet_by_days(lat, lon, year, days):
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
        daymet_data = pd.read_csv(StringIO(response.text), skiprows=6)
        return daymet_data[daymet_data['yday'].isin(days)]
    else:
        print(f"Error fetching data for {lat}, {lon} in year {year}: {response.status_code}")
        return None

def load_senescence_data(file_path):
    plant_data = pd.read_csv(file_path)
    plant_data = plant_data.dropna(subset=['Tos'])

    site_data = {
        'Toolik': (68.623, -149.606, 2362),
        'Coldfoot': (67.25, -150.175, 1014),
        'Sagwon': (69.373, -148.700, 675),
        'TL': (68.623, -149.606, 2362),
        'CF': (67.25, -150.175, 1014),
        'SG': (69.373, -148.700, 675)
    }

    plant_data['current_lat'] = plant_data['Site'].map(lambda site: site_data[site][0] if site in site_data else None)
    plant_data['current_long'] = plant_data['Site'].map(lambda site: site_data[site][1] if site in site_data else None)
    plant_data['current_altitude'] = plant_data['Site'].map(lambda site: site_data[site][2] if site in site_data else None)

    plant_data['origin_lat'] = plant_data['Src'].map(lambda src: site_data[src][0] if src in site_data else None)
    plant_data['origin_long'] = plant_data['Src'].map(lambda src: site_data[src][1] if src in site_data else None)
    plant_data['origin_altitude'] = plant_data['Src'].map(lambda src: site_data[src][2] if src in site_data else None)

    # Drop legacy columns
    for col in ['lat', 'long', 'altitude']:
        if col in plant_data.columns:
            plant_data.drop(columns=col, inplace=True)

    return plant_data

def compile_weather_data(senescence_df, hourly_weather_path=None):
    print("Compiling weather data...")

    daymet_cache = {}
    results = []

    for _, row in tqdm(senescence_df.iterrows(), total=senescence_df.shape[0]):
        site, src, plot, ind, yrm, gr, tcode, tos = (
            row["Site"], row["Src"], row["Plot"], row["Ind"], row["Yrm"], row["gr"], row["Tcode"], row["Tos"]
        )
        current_lat, current_long, current_altitude = (
            row["current_lat"], row["current_long"], row["current_altitude"]
        )
        origin_lat, origin_long, origin_altitude = (
            row["origin_lat"], row["origin_long"], row["origin_altitude"]
        )

        cache_key = (current_lat, current_long, yrm)
        if cache_key not in daymet_cache:
            daymet_data = fetch_daymet_by_days(current_lat, current_long, yrm, days=range(1, 367))
            if daymet_data is not None:
                daymet_cache[cache_key] = daymet_data
            else:
                continue
        else:
            daymet_data = daymet_cache[cache_key]

        daymet_data = daymet_data[(daymet_data['yday'] >= 150) & (daymet_data['yday'] <= 250)]
        daymet_data = daymet_data.copy()
        daymet_data['daily_rad'] = daymet_data['dayl (s)'] * daymet_data['srad (W/m^2)']

        daymet_data["Site"] = site
        daymet_data["Src"] = src
        daymet_data["Plot"] = plot
        daymet_data["Ind"] = ind
        daymet_data["Yrm"] = yrm
        daymet_data["Tcode"] = tcode
        daymet_data["gr"] = gr
        daymet_data["days_until_senescence"] = tos - daymet_data["yday"]
        daymet_data["doy"] = daymet_data["yday"].astype(int)

        # Add both current and origin coordinates
        daymet_data["current_lat"] = current_lat
        daymet_data["current_long"] = current_long
        daymet_data["current_altitude"] = current_altitude
        daymet_data["origin_lat"] = origin_lat
        daymet_data["origin_long"] = origin_long
        daymet_data["origin_altitude"] = origin_altitude

        results.append(daymet_data)

    if not results:
        print("No weather data compiled.")
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)
    final_df.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", "yday"], inplace=True)

    if hourly_weather_path is not None:
        hourly = pd.read_csv(hourly_weather_path)
        hourly['ts'] = pd.to_datetime(hourly['ts'])
        hourly['year'] = hourly['ts'].dt.year
        hourly['doy'] = hourly['ts'].dt.dayofyear

        def agg_day(df, treatment):
            temps = df['temp'].copy()
            if treatment == 'warming':
                temps += 0.5

            gdh = temps[temps > 0].sum()
            frost_hours = (temps < 0).sum()

            snow_amt = df['snow'].sum() if 'snow' in df else pd.NA
            min_sea = df['sea'].min() if 'sea' in df else pd.NA
            max_sea = df['sea'].max() if 'sea' in df else pd.NA

            # return pd.Series({
            #     'growing_degree_hours': gdh,
            #     'frost_hours': frost_hours,
            #     'day_length': day_length * 3600,
            #     'snow_amount': snow_amt,
            #     'min_sea': min_sea,
            #     'max_sea': max_sea,
            #     'PTU': ptu
            # })
        
            return pd.Series({
                'growing_degree_hours': gdh,
                'frost_hours': frost_hours,
                'snow_amount': snow_amt,
                'min_sea': min_sea,
                'max_sea': max_sea,
            })
        

        site_map = {'TL': 'Toolik', 'CF': 'Coldfoot', 'SG': 'Sagwon'}
        hourly['site'] = hourly['site'].replace(site_map)

        daily = hourly.groupby(['site', 'year', 'doy']).apply(
            lambda x: agg_day(x, row['gr'])
        ).reset_index()

        final_df = final_df.reset_index()
        merged = final_df.merge(
            daily, left_on=['Site', 'Yrm', 'doy'], right_on=['site', 'year', 'doy'], how='left'
        )

        for col in ['site', 'year', 'year_x', 'year_y']:
            if col in merged.columns:
                merged.drop(columns=col, inplace=True)

        final_df = merged.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", 'gr', "yday"])
        
    def add_interactions(df):
        df = df.copy()

        # Frost × photoperiod
        df['frost_x_daylen'] = df['frost_hours'] * df['day_length']

        # Frost × thermal accumulation
        if 'growing_degree_hours' in df:
            df['frost_x_gdh'] = df['frost_hours'] * df['growing_degree_hours']

        # Light × temperature
        df['daylen_x_tmax'] = df['day_length'] * df['tmax (deg c)']
        df['daylen_x_tmin'] = df['day_length'] * df['tmin (deg c)']
        df['min_sea_x_max_sea'] = df['min_sea'] * df['max_sea']
        df['min_sea_x_frost'] = df['min_sea'] * df['frost_hours']
        df['GDH_x_max_sea'] = df['growing_degree_hours'] * df['max_sea']
        df['tmin_x_min_sea'] = df['tmin (deg c)'] * df['max_sea']

        # Snow × frost
        if 'snow_amount' in df:
            df['snow_x_frost'] = df['snow_amount'] * df['frost_hours']
        if 'swe (kg/m^2)' in df:
            df['swe_x_tmin'] = df['swe (kg/m^2)'] * df['tmin (deg c)']

        # Thermal × daylength
        if 'PTU' in df:
            df['ptu_x_daylen'] = df['PTU'] * df['day_length']
            df['ptu_x_frost'] = df['PTU'] * df['frost_hours']

        return df
    
    df_reset = final_df.reset_index()
    day_length = df_reset['dayl (s)'] / 3600

    day_length = np.where(
        df_reset['gr'] == 'shading', np.maximum(0, day_length - 1),
        np.where(df_reset['gr'] == 'lighting', np.minimum(24, day_length + 1), day_length)
    )
    df_reset['day_length'] = day_length

    final_df = df_reset.set_index(["Site", "Src", "Plot", "Ind", "Yrm", "Tcode", "gr", "yday"])

    final_df = add_interactions(final_df)

    return final_df

def plot_variables(df):
    exclude_vars = ['Site', 'Src', 'Plot', 'Ind', 'Yrm', 'Tcode', 'gr', 'yday', 'days_until_senescence']
    df = df.reset_index()
    df['group'] = df['Site'] + '-' + df['Yrm'].astype(str) + '-' + df['gr'].fillna('control')
    plot_vars = [col for col in df.columns if col not in exclude_vars]

    os.makedirs('env_plots', exist_ok=True)

    for var in plot_vars:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='doy', y=var, hue='group')
        plt.title(f'{var} over Growing Season')
        plt.xlabel('Day of Year')
        plt.ylabel(var)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        safe_filename = var.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_") \
            .replace("^2", "squared").replace("^", "_pow_").replace(".", "_").replace(",", "_")
        plt.savefig(f'env_plots/{safe_filename}.png')
        plt.close()

def main():
    args = parse_arguments()
    tos_df = load_senescence_data(args.input)
    print(tos_df['gr'].unique())
    compiled_weather_data = compile_weather_data(tos_df, hourly_weather_path='data/AS25_WSdata_GS.csv')
    compiled_weather_data.to_csv(args.output)
    print(compiled_weather_data.columns)
    print('done')
    # plot_variables(compiled_weather_data)

if __name__ == "__main__":
    main()

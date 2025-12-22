import argparse
import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Determine Time of Senescence (ToS) from phenological data.")
    parser.add_argument("-i", "--input", required=True, help="Input file containing phenological data.")
    parser.add_argument("-o", "--output", required=True, help="Output file to save ToS results.")
    return parser.parse_args()


def clean_data(df):

    # df = df.drop(columns=['Yrt', 'tr', 'Rep'])
    # df = df[~df['site'].isin(['Imnaviat', 'Chamber', 'Greenhouse'])]
    # df = df[~df['Yrm'].isin([1985,1986])]
    # df = df[~df['gr'].isin(['fertilizing','C','L','S'])]


    # Count measurements per tiller
    tiller_measurements = df.groupby(['Src','Plot','Ind','Tiller','site','Yrm'])['doy'].nunique().reset_index()
    print("\nTiller measurement counts:")
    print(tiller_measurements)
    
    # Filter out tillers that don't make it halfway through season
    year_midpoints = df.groupby('Yrm')['doy'].median()
    df = df.merge(
        df.groupby(['site', 'Plot', 'Ind', 'Tiller', 'Yrm', 'gr'])['doy'].max().reset_index(),
        on=['site', 'Plot', 'Ind', 'Tiller', 'Yrm', 'gr']
    )
    df = df[df['doy_y'] >= df['Yrm'].map(year_midpoints)]
    df = df.drop(columns=['doy_y'])
    
    df = df.rename(columns={'doy_x': 'doy'})

    leaf_counts = df.groupby(['Src','Plot','Ind','Tiller','Leaf','site','Yrm'])['doy'].count().rename('leaf_count').reset_index()
    tussock_counts = df.groupby(['Src','site','Plot','Ind','Yrm'])['doy'].count().rename('tussock_count').reset_index()

    
    merged_counts = leaf_counts.merge(tussock_counts, on=['site', 'Src','Plot', 'Ind', 'Yrm', 'gr'])
    valid_leaves = merged_counts[merged_counts['leaf_count'] > 3]
    
    df = df.merge(valid_leaves[['site', 'Plot', 'Ind', 'Leaf', 'Yrm', 'gr']], on=['site', 'Plot', 'Ind', 'Leaf', 'Yrm', 'gr'])

    # --- Make a copy of cleaned tiller-level data before aggregation ---
    df_cleaned = df.copy()

    # Remove leaf measurement spikes: if a leaf measurement spikes up or down by >=50 and then returns to roughly its previous value, remove the spike
    def remove_spikes(leaf_df):
        leaf_df = leaf_df.sort_values('doy').copy()
        gl = leaf_df['gl'].values
        mask = np.ones(len(gl), dtype=bool)
        for i in range(1, len(gl) - 1):
            prev_val = gl[i - 1]
            curr_val = gl[i]
            next_val = gl[i + 1]
            # If current value is a spike (diff >= 40 from prev and returns to prev at next)
            if abs(curr_val - prev_val) >= 40 and abs(next_val - prev_val) < 10:
                mask[i] = False
        return leaf_df[mask]

    df = df.groupby(['site', 'Src','Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm', 'gr'], group_keys=False).apply(remove_spikes).reset_index(drop=True)

    # Remove entire leaf if it drops >70% from previous value and does not return next observation
    def remove_leaves_with_large_drop(leaf_df):
        leaf_df = leaf_df.sort_values('doy').copy()
        gl = leaf_df['gl'].values
        for i in range(1, len(gl) - 1):
            prev_val = gl[i - 1]
            curr_val = gl[i]
            next_val = gl[i + 1]
            if prev_val > 0 and (curr_val < 0.3 * prev_val) and (next_val < 0.3 * prev_val):
                # Drop the entire leaf
                return pd.DataFrame([], columns=leaf_df.columns)
        return leaf_df

    df = df.groupby(['site', 'Src','Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm', 'gr'], group_keys=False).apply(remove_leaves_with_large_drop).reset_index(drop=True)

    # Aggregate to total green length (gl) per tiller per day
    tiller_gl = (
        df.groupby(['site','Src','Plot', 'Ind', 'Tiller', 'doy', 'Yrm', 'gr'])
        .agg({'gl': 'sum'})
        .reset_index()
        .rename(columns={'Ind': 'Tussock', 'gl': 'gl_total'})
    )

    # Interpolate and smooth at the tiller level
    interpolated = []
    for (site, src, plot, tussock, tiller, yrm, gr), group in tiller_gl.groupby(['site','Src','Plot', 'Tussock', 'Tiller', 'Yrm', 'gr']):
        group = group.sort_values(by='doy').drop_duplicates(subset='doy').set_index('doy')
        # Interpolate daily values
        doy_range = np.arange(group.index.min(), group.index.max() + 1)
        group = group.reindex(doy_range)
        group['gl_total'] = group['gl_total'].interpolate()
        # Apply moving average smoothing to total green length
        group['gl_smoothed'] = group['gl_total'].rolling(window=14, center=True, min_periods=1).mean()
        group[['site','Src','Plot', 'Tussock', 'Tiller', 'Yrm', 'gr']] = site, src, plot, tussock, tiller, yrm, gr
        interpolated.append(group.reset_index())

    tiller_gl_smoothed = pd.concat(interpolated, ignore_index=True)

    return tiller_gl_smoothed
    # -------------------------------------------------------------------------


def detect_senescence(df, params=None):
    print(11111)
    print(df)
    """
    Detect senescence by finding first sustained drop below threshold after peak green length.
    """
    if params is None:
        params = {
            'window_size': 7,  # This is used as sustained_days
            'threshold_fraction': 0.7
        }

    senescence_records = []

    for data in df.groupby(['site','Src','Plot','Tussock','Tiller','Yrm','gr']):
        site_plot_tussock_tiller, group = data
        group = group.sort_values(by='doy').set_index('doy')

        # Find peak and calculate threshold
        peak_doy = group['gl_smoothed'].idxmax()
        peak_value = group['gl_smoothed'].max()
        drop_threshold = peak_value * params['threshold_fraction']

        # Find days that drop below threshold after peak
        drop_days = group[(group.index > peak_doy) & 
                         (group['gl_smoothed'] < drop_threshold)].index

        # Check for sustained drops
        sustained_drop = []
        if len(drop_days) > params['window_size']:
            sustained_drop = [
                day for i, day in enumerate(drop_days[:-params['window_size']])
                if all(group['gl_smoothed'].loc[day:day + params['window_size']] < drop_threshold)
            ]

        threshold_day = sustained_drop[0] if sustained_drop else None



        senescence_records.append({
            'Site': site_plot_tussock_tiller[0],
            'Src': site_plot_tussock_tiller[1],
            'Plot': site_plot_tussock_tiller[2],
            'Tussock': site_plot_tussock_tiller[3],
            'Tiller':site_plot_tussock_tiller[4],
            'Yrm': site_plot_tussock_tiller[5],
            'gr': site_plot_tussock_tiller[6],
            'Tos': threshold_day
        })

    return pd.DataFrame(senescence_records).dropna(subset=['Tos'])


def main():
    args = parse_arguments()
    data = pd.read_csv(args.input)
    data = clean_data(data)

    # Use tiller_cleaned (leaf-level) for leaf filtering
    # data = filter_leaves(tiller_cleaned)

    # data = interpolate_data(data)

    params = {
    "window_size": 5,
    "threshold_fraction": 0.9,
    "min_drop_duration": 7,
    "offset": -2
    }

    data = detect_senescence(data, params=params)
    print(data)

    data = data.rename(columns={'Tussock': 'Ind'})

    # Create Tcode column
    data['Tcode'] = data.apply(lambda row: f"{row['Site']}.{row['Plot']}.{row['Ind']}.{row['Tiller']}.{row['Yrm']}", axis=1)
    print(data)

    data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
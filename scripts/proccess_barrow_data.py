import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def process_single_csv(path):
    """Load one CSV, filter & transform, return evag_gl dataframe."""
    df = pd.read_csv(path)

    # Make sure numeric fields are numeric
    df["strJulianDay"] = pd.to_numeric(df["strJulianDay"], errors="coerce")
    df["numResponse"] = pd.to_numeric(df["numResponse"], errors="coerce")

    # ====== 2. Filter: only EVAG & GROW ======
    mask = (df["strGSpp"] == "EVAG") & (df["strDataType"] == "GROW")
    evag = df[mask].copy()

    if evag.empty:
        return pd.DataFrame(columns=["Src", "Plot", "Ind", "Tiller", "Leaf", "doy", "gl", "Site", "Yrm"])

    # >>> Zero out negative response values before summing <<<
    evag["numResponse"] = evag["numResponse"].clip(lower=0)

    # ====== 3. Build new columns ======
    #   Src  = strSitCom
    #   Plot = strPlotName (so different plots can be distinguished)
    #   Site = strSitCom
    evag["Src"] = evag["strSitCom"]
    evag["Plot"] = evag["strPlotName"]   # change to evag["strSitCom"] if you prefer
    evag["Ind"] = evag["strPlantID"]
    evag["Tiller"] = evag["strPlantID"]
    evag["Leaf"] = 1
    evag["doy"] = evag["strJulianDay"]
    evag["Site"] = evag["strSitCom"]
    evag["Yrm"] = evag["strYear"]

    # ====== 4. Aggregate gl (sum per individual per date) ======
    group_cols = ["Src", "Plot", "Ind", "Tiller", "Leaf", "doy", "Site", "Yrm"]

    evag_gl = (
        evag
        .groupby(group_cols, as_index=False)["numResponse"]
        .sum()
        .rename(columns={"numResponse": "gl"})
    )

    # Reorder columns to exact header order
    evag_gl = evag_gl[["Src", "Plot", "Ind", "Tiller", "Leaf", "doy", "gl", "Site", "Yrm"]]

    return evag_gl


def main():
    parser = argparse.ArgumentParser(description="Process multiple CSVs into a single EVAG GROW leaf-length file and plot.")
    parser.add_argument(
        "csvs",
        nargs="+",
        help="List of input CSV files"
    )
    parser.add_argument(
        "--out_csv",
        default="EVAG_GROW_gl_by_ind_merged.csv",
        help="Output CSV filename for merged results"
    )
    parser.add_argument(
        "--plot_path",
        default="plots/BG_AG_gl_timeseries_subplots_merged.png",
        help="Output PNG filename for the merged plot"
    )
    args = parser.parse_args()

    all_evag_gl = []

    for path in args.csvs:
        print(f"Processing {path}...")
        evag_gl = process_single_csv(path)
        if not evag_gl.empty:
            all_evag_gl.append(evag_gl)
        else:
            print(f"  (No EVAG / GROW rows found in {path})")

    if not all_evag_gl:
        print("No EVAG / GROW records found in any input CSVs. Nothing to save or plot.")
        return

    # Concatenate everything into one big dataframe
    merged = pd.concat(all_evag_gl, ignore_index=True)

    # Save merged CSV
    merged.to_csv(args.out_csv, index=False)
    print(f"Saved merged CSV → {args.out_csv}")

    # ====== Plot from merged dataframe ======
    os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)

    groups = merged.groupby(["Plot", "Ind", "Yrm"])
    n_groups = len(groups)

    if n_groups == 0:
        print("No EVAG / GROW records found to plot in merged data.")
        return

    ncols = 4
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True,
        sharey=True,
    )

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, ((plot, ind, yr), g) in zip(axes, groups):
        g = g.sort_values("doy")
        ax.plot(g["doy"], g["gl"], marker="o")
        ax.set_title(f"{plot}-{ind}-{yr}")
        ax.set_xlabel("doy")
        ax.set_ylabel("gl")

    # Hide any unused axes
    for ax in axes[n_groups:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved merged plot → {args.plot_path}")


if __name__ == "__main__":
    main()

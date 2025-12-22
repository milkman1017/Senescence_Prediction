#!/usr/bin/env python3
import os, json, argparse, math
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from joblib import load
from sklearn.preprocessing import LabelEncoder

print("shap version:", shap.__version__)  # should be >= 0.28

from keras import config as keras_config  # optional

keras_config.enable_unsafe_deserialization()

# -----------------------------
# Helpers: windowing + encoding
# -----------------------------
def rebuild_label_encoders(ce_spec):
    """Recreate LabelEncoder objects from saved classes_."""
    encoders = {}
    for col, meta in ce_spec.items():
        le = LabelEncoder()
        le.classes_ = np.array(meta["classes_"])
        encoders[col] = le
    return encoders

def apply_categorical_encoders(df, static_cols, encoders):
    """Encode object/string static columns using saved LabelEncoders."""
    for col in static_cols:
        if col in encoders:
            le = encoders[col]
            # map unknowns to -1 (consistent with your training logic for 2024)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            # if not encoded at train time (numeric), leave as is
            pass
    return df

def create_windows(df, weather_cols, static_cols, time_interval):
    """Replicates your create_windows() from model.py."""
    windows, labels, statics, years, groups = [], [], [], [], []
    # derive grouping columns like in your code
    grouping_cols = [c for c in static_cols if c not in ['Site_lat', 'Site_long']]
    grouping_cols += ['Plot', 'Ind', 'Yrm', 'Tcode', 'gr']
    grouping_cols = list(dict.fromkeys(grouping_cols))

    grouped = df.groupby(grouping_cols, sort=False)
    for key, g in grouped:
        g = g.sort_values('yday').reset_index(drop=True)
        for i in range(len(g) - time_interval + 1):
            w = g.iloc[i:i+time_interval]
            windows.append(w.copy())
            labels.append(w['days_until_senescence'].iloc[-1])
            statics.append(w.iloc[-1][static_cols].astype(float).values)
            years.append(int(w['Yrm'].iloc[-1]))
            groups.append("_".join(map(str, key)))
    return windows, np.array(labels), np.array(statics), np.array(years), np.array(groups)

def windows_to_arrays(windows, weather_cols, static_cols):
    Xw = np.array([w[weather_cols].values for w in windows])   # (N, T, Fw)
    Xs = np.array([w.iloc[-1][static_cols].astype(float).values for w in windows])  # (N, Fs)
    y  = np.array([w['days_until_senescence'].iloc[-1] for w in windows], dtype=float)
    return Xw, Xs, y

# -----------------------------
# Plotting helpers
# -----------------------------
def heatmap_time_feature(mean_abs_shap_3d, weather_cols, out_png):
    """
    mean_abs_shap_3d: (T, Fw) mean(|SHAP|) aggregated across samples
    Writes a heatmap of time (rows) × weather feature (cols).
    """
    plt.figure(figsize=(1.2*len(weather_cols), 0.5*mean_abs_shap_3d.shape[0]))
    plt.imshow(mean_abs_shap_3d, aspect="auto", interpolation="nearest")
    plt.yticks(range(mean_abs_shap_3d.shape[0]), [f"t-{i}" for i in range(mean_abs_shap_3d.shape[0])])
    plt.xticks(range(len(weather_cols)), weather_cols, rotation=45, ha="right")
    plt.colorbar(label="mean |SHAP|")
    plt.title("Weather importance over time × feature")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="SHAP analysis for senescence TF/Keras model (two-input).")
    ap.add_argument("--model_dir", default="models/senescence_model_lead", help="SavedModel directory or .h5")
    ap.add_argument("--data_csv", required=True, help="Same raw CSV you trained on (or a subset).")
    ap.add_argument("--artifacts_json", default="models/artifacts.json", help="Path to saved artifacts.json")
    ap.add_argument("--scalers_joblib", default="models/scalers.joblib", help="Path to saved scalers.joblib")
    ap.add_argument("--subset_rows", type=int, default=None, help="Optional: only read first N rows from CSV")
    ap.add_argument("--max_bg", type=int, default=200, help="Background sample size for SHAP")
    ap.add_argument("--max_windows", type=int, default=3000, help="Limit number of windows to explain (speed)")
    ap.add_argument("--outdir", default="shap_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load model + preprocess artifacts
    print("[load] model…")
    model = tf.keras.models.load_model(args.model_dir, compile=False)

    print("[load] artifacts…")
    with open(args.artifacts_json) as f:
        art = json.load(f)
    weather_cols = art["weather_columns"]
    static_cols  = art["static_columns"]
    time_interval = int(art["time_interval"])
    label_encoders = rebuild_label_encoders(art.get("categorical_encoders", {}))

    print("[load] scalers…")
    scalers = load(args.scalers_joblib)
    weather_scaler = scalers["weather_scaler"]
    static_scaler  = scalers["static_scaler"]
    target_scaler  = scalers["target_scaler"]  # not used for SHAP itself

    # 2) Load CSV and light cleaning to match training
    print("[data] reading csv…")
    df = pd.read_csv(args.data_csv, nrows=args.subset_rows)
    df.columns = df.columns.str.strip()

    # Recreate lat/long mapping if present in static cols (matches your train code)
    lat_long = {
        'Toolik': (68.623, -149.606),'TL': (68.623, -149.606),
        'Coldfoot': (67.25, -150.175),'CF': (67.25, -150.175),
        'Sagwon': (69.373, -148.700),'SG': (69.373, -148.700)
    }
    # Expand Site/Src to lat/long if they are expected
    if "Site_lat" in static_cols and "Site_long" in static_cols and "Site" in df.columns:
        df["Site_lat"]  = df["Site"].map(lambda x: lat_long.get(x, (0,0))[0])
        df["Site_long"] = df["Site"].map(lambda x: lat_long.get(x, (0,0))[1])
    if "Src_lat" in static_cols and "Src_long" in static_cols and "Src" in df.columns:
        df["Src_lat"]  = df["Src"].map(lambda x: lat_long.get(x, (0,0))[0])
        df["Src_long"] = df["Src"].map(lambda x: lat_long.get(x, (0,0))[1])

    # Yrm as string (as in your loader) and dedupe keys
    if "Yrm" in df.columns:
        df["Yrm"] = df["Yrm"].astype(str)
    if set(['Site','Src','Plot','Ind','Yrm','Tcode','gr','yday']).issubset(df.columns):
        df = df.drop_duplicates(subset=['Site','Src','Plot','Ind','Yrm','Tcode','gr','yday'], keep='first')

    # ensure target exists (only used for plotting/comparison; SHAP uses features)
    if "days_until_senescence" not in df.columns:
        raise ValueError("CSV must contain 'days_until_senescence' for window labels (even if not used in SHAP).")

    # Re-encode categoricals exactly as in training
    df = apply_categorical_encoders(df, static_cols, label_encoders)

    # 3) Build rolling windows and arrays
    print("[windows] building windows…")
    windows, labels, statics, years, groups = create_windows(df, weather_cols, static_cols, time_interval)
    if len(windows) == 0:
        raise RuntimeError("No windows built — check that your CSV has enough consecutive days per group.")

    # Convert to arrays
    Xw_raw, Xs_raw, y_raw = windows_to_arrays(windows, weather_cols, static_cols)

    # Optional cap for speed
    n = min(args.max_windows, Xw_raw.shape[0])
    Xw_raw, Xs_raw, y_raw = Xw_raw[:n], Xs_raw[:n], y_raw[:n]

    # 4) Scale inputs using training scalers
    print("[scale] applying training scalers…")
    # weather: scale per-feature then reshape back
    ws = Xw_raw.reshape(-1, Xw_raw.shape[-1])
    ws = weather_scaler.transform(ws)
    Xw = ws.reshape(Xw_raw.shape)

    # static
    Xs = static_scaler.transform(Xs_raw)

    # 5) Prepare SHAP background (representative subset)
    rng = np.random.default_rng(0)
    idx_bg = rng.choice(Xw.shape[0], size=min(args.max_bg, Xw.shape[0]), replace=False)
    bg_weather = Xw[idx_bg]   # (Bbg, T, Fw)
    bg_static  = Xs[idx_bg]   # (Bbg, Fs)

    # 6) Build SHAP explainer (KernelExplainer over flattened inputs)
    # --------------------------------------------------------------
    # Flatten helpers
    T, Fw = Xw.shape[1], Xw.shape[2]
    Fs = Xs.shape[1]

    def flatten_inputs(xw, xs):
        # xw: (N, T, Fw) -> (N, T*Fw); xs: (N, Fs)
        return np.concatenate([xw.reshape(xw.shape[0], T*Fw), xs], axis=1)

    def split_flat(Xflat):
        # inverse of flatten_inputs for batches inside predict_fn
        xw_flat = Xflat[:, :T*Fw]
        xs_flat = Xflat[:, T*Fw:]
        xw = xw_flat.reshape(-1, T, Fw)
        xs = xs_flat.reshape(-1, Fs)
        return [xw, xs]

    # Background in flat space
    bg_flat = flatten_inputs(bg_weather, bg_static)   # (Bbg, T*Fw+Fs)

    # Prediction function that maps flat -> model(list) -> 1D
    def predict_fn(Xflat):
        xw, xs = split_flat(np.array(Xflat, dtype=np.float32))
        # model outputs shape (N,1) or (N,), make it (N,)
        y = model.predict([xw, xs], verbose=0)
        return y.reshape(-1)

    # 7) Compute SHAP values with KernelExplainer
    # -------------------------------------------
    # You can tune nsamples for speed/accuracy (default is ~O(D^2)):
    nsamples = min(1000, 2*(T*Fw + Fs))  # reasonable default; make a CLI arg if you like

    explainer = shap.KernelExplainer(predict_fn, bg_flat)
    Xflat = flatten_inputs(Xw, Xs)

    # Chunk the explanation to control runtime/memory
    # (Change chunk_size to taste; smaller = less RAM, longer runtime)
    chunk_size = 512
    all_shap = []
    for i in range(0, Xflat.shape[0], chunk_size):
        batch = Xflat[i:i+chunk_size]
        # l1_reg="aic" helps sparsify attributions for tabular-like inputs
        sv = explainer.shap_values(batch, nsamples=nsamples, l1_reg="aic")
        # shap_values returns (batch, D)
        all_shap.append(sv)
    vals_flat = np.vstack(all_shap)             # (N, D)

    # Reshape back to weather/static tensors
    vals_weather = vals_flat[:, :T*Fw].reshape(-1, T, Fw)   # (N, T, Fw)
    vals_static  = vals_flat[:, T*Fw:]                      # (N, Fs)
    
    explainer_type = "kernel"

    # 8) Save numeric outputs
    np.save(os.path.join(args.outdir, "shap_weather.npy"), vals_weather)
    np.save(os.path.join(args.outdir, "shap_static.npy"), vals_static)
    pd.DataFrame({
        "n_samples": [Xw.shape[0]],
        "time_steps": [Xw.shape[1]],
        "n_weather_features": [Xw.shape[2]],
        "n_static_features": [Xs.shape[1]],
        "explainer": [explainer_type],
    }).to_csv(os.path.join(args.outdir, "summary_meta.csv"), index=False)

    # 9) Global importance for STATIC features
    mean_abs_static = np.abs(vals_static).mean(axis=0)  # (Fs,)
    imp_static = pd.DataFrame({"feature": static_cols, "mean_abs_shap": mean_abs_static}) \
                    .sort_values("mean_abs_shap", ascending=False)
    imp_static.to_csv(os.path.join(args.outdir, "importance_static.csv"), index=False)

    # 10) Global importance for WEATHER across time and features
    # a) feature-wise (aggregate across time and samples)
    mean_abs_weather_feat = np.abs(vals_weather).mean(axis=(0, 1))  # (Fw,)
    imp_weather_feat = pd.DataFrame({"feature": weather_cols, "mean_abs_shap": mean_abs_weather_feat}) \
                        .sort_values("mean_abs_shap", ascending=False)
    imp_weather_feat.to_csv(os.path.join(args.outdir, "importance_weather_by_feature.csv"), index=False)

    # b) time-wise (aggregate across features)
    mean_abs_weather_time = np.abs(vals_weather).mean(axis=(0, 2))  # (T,)
    pd.DataFrame({"t_index": np.arange(mean_abs_weather_time.shape[0]),
                  "mean_abs_shap": mean_abs_weather_time}) \
        .to_csv(os.path.join(args.outdir, "importance_weather_by_time.csv"), index=False)

    # c) full heatmap time × feature
    heatmap = np.abs(vals_weather).mean(axis=0)  # (T, Fw)
    heatmap_time_feature(heatmap, weather_cols, os.path.join(args.outdir, "weather_time_feature_heatmap.png"))

        
    # 11) PLOTS (Kernel SHAP version — no shap_exp object needed)
    # 11a) Static bar plot
    try:
        plt.figure(figsize=(max(6, 0.4*len(static_cols)), 0.5*len(static_cols)))
        order = np.argsort(-mean_abs_static)
        plt.barh(np.array(static_cols)[order], mean_abs_static[order])
        plt.gca().invert_yaxis()
        plt.xlabel("mean |SHAP|")
        plt.title("Static feature importance")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "static_bar.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    # 11b) Optional: a simple “beeswarm-like” scatter for static (signed SHAP)
    #       (not as fancy as shap.plots.beeswarm but conveys direction)
    try:
        topk = 20
        top_features = np.array(static_cols)[order][:topk]
        top_idx = [static_cols.index(f) for f in top_features]
        plt.figure(figsize=(8, max(4, 0.4*len(top_features))))
        yticklabels = []
        for i, j in enumerate(top_idx):
            y = np.full(vals_static.shape[0], i)
            x = vals_static[:, j]  # signed SHAP
            plt.plot(x, y, 'o', markersize=2, alpha=0.4)
            yticklabels.append(top_features[i])
        plt.yticks(range(len(top_features)), yticklabels)
        plt.axvline(0, ls='--', lw=1)
        plt.xlabel("SHAP (signed)")
        plt.title("Static feature SHAP (top 20)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "static_beeswarm.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    # 11c) Weather: per-feature temporal importance (average |SHAP| over samples)
    try:
        top_weather = imp_weather_feat["feature"].tolist()[:5]
        t = np.arange(Xw.shape[1])
        for feat in top_weather:
            j = weather_cols.index(feat)
            series = np.abs(vals_weather[:, :, j]).mean(axis=0)
            plt.figure()
            plt.plot(t, series)
            plt.xlabel("time index within window")
            plt.ylabel(f"|SHAP| (avg) for {feat}")
            plt.title(f"Temporal importance: {feat}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"weather_temporal_{feat}.png"), dpi=200)
            plt.close()
    except Exception:
        pass

        print(f"[OK] Wrote SHAP outputs → {args.outdir}")

if __name__ == "__main__":
    main()

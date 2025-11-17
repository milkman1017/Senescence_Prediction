import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, ReLU, LayerNormalization, Lambda,
                                     BatchNormalization, Concatenate, Bidirectional, LSTM, MultiHeadAttention, Add, GlobalMaxPooling1D, GlobalAveragePooling1D, Reshape)
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import math
from tensorflow.keras import regularizers
from joblib import dump
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model to predict days until senescence.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input data file (CSV).')
    parser.add_argument('--time_interval', type=int, default=7,
                      help='Number of consecutive days used as input (window length).')
    parser.add_argument('--subset_size', type=int, default=None,
                      help='If set, only load this many rows (for quick testing).')
    parser.add_argument('--weather_columns', type=str, nargs='+', required=True,
                      help='List of column names for weather features (e.g., tmax tmin precip)')
    parser.add_argument('--static_columns', type=str, nargs='+', required=True,
                      help='List of column names for static features (e.g., Site Src Plot)')
    parser.add_argument('--stride', type=int, default=1,
                      help='N windows to skip to avoid too many almost identical windows')


    return parser.parse_args()

def load_data(args):
    """
    Load the CSV, engineer features, split into pre-2024 and 2024,
    and build rolling windows for the model.
    """
    df = pd.read_csv(args.file_path)
    df.columns = df.columns.str.strip()

    # Drop exact duplicates based on key identifying columns
    df = df.drop_duplicates(
        subset=['Site', 'Src', 'Plot', 'Ind', 'Yrm', 'Tcode', 'gr', 'yday'],
        keep='first'
    )

    # ------------------------------------------------------------------
    # Static lat/long encodings for Site and Src
    # ------------------------------------------------------------------
    lat_long = {
        'Toolik':   (68.623, -149.606),
        'TL':       (68.623, -149.606),
        'Coldfoot': (67.25,  -150.175),
        'CF':       (67.25,  -150.175),
        'Sagwon':   (69.373, -148.700),
        'SG':       (69.373, -148.700),
    }

    if 'Site' in args.static_columns:
        df['Site_lat']  = df['Site'].map(lambda x: lat_long.get(x, (0, 0))[0])
        df['Site_long'] = df['Site'].map(lambda x: lat_long.get(x, (0, 0))[1])
        args.static_columns.remove('Site')
        args.static_columns.extend(['Site_lat', 'Site_long'])

    if 'Src' in args.static_columns:
        df['Src_lat']  = df['Src'].map(lambda x: lat_long.get(x, (0, 0))[0])
        df['Src_long'] = df['Src'].map(lambda x: lat_long.get(x, (0, 0))[1])
        args.static_columns.remove('Src')
        args.static_columns.extend(['Src_lat', 'Src_long'])

    # ------------------------------------------------------------------
    # Ensure Yrm is string and split pre-2024 vs 2024
    # ------------------------------------------------------------------
    if 'Yrm' in df.columns:
        df['Yrm'] = df['Yrm'].astype(str)

    if 'Yrm' in df.columns:
        df_2024 = df[df['Yrm'].str.startswith('2024')].copy()
        df_train = df[~df['Yrm'].str.startswith('2024')].copy()
    else:
        df_2024 = pd.DataFrame()
        df_train = df.copy()

    # Optional row subsampling for quick tests
    if args.subset_size:
        df_train = df_train.head(args.subset_size)
        if not df_2024.empty:
            df_2024 = df_2024.head(args.subset_size)

    # ------------------------------------------------------------------
    # Target presence check
    # ------------------------------------------------------------------
    if 'days_until_senescence' not in df.columns:
        raise ValueError("Missing target column 'days_until_senescence'")

    # ------------------------------------------------------------------
    # Day-of-year cyclical encoding: use only sin/cos, no raw doy/yday
    # ------------------------------------------------------------------
    # Determine which column in the data holds day-of-year
    doy_source_col = None
    for cand in ['yday', 'doy']:
        if cand in df.columns:
            doy_source_col = cand
            break

    # Did the user ask to use a day-of-year feature?
    wants_doy = any(
        c in (args.weather_columns + args.static_columns)
        for c in ['doy', 'yday']
    )

    if doy_source_col is not None and wants_doy:
        tau = 2 * np.pi

        # Full DF
        phase = (df[doy_source_col].astype(float) % 365.25) / 365.25
        df['doy_sin'] = np.sin(tau * phase)
        df['doy_cos'] = np.cos(tau * phase)

        # Train subset
        phase_train = (df_train[doy_source_col].astype(float) % 365.25) / 365.25
        df_train['doy_sin'] = np.sin(tau * phase_train)
        df_train['doy_cos'] = np.cos(tau * phase_train)

        # 2024 subset (if present)
        if not df_2024.empty:
            phase_2024 = (df_2024[doy_source_col].astype(float) % 365.25) / 365.25
            df_2024['doy_sin'] = np.sin(tau * phase_2024)
            df_2024['doy_cos'] = np.cos(tau * phase_2024)

        # Remove raw doy / yday from both feature lists
        for col_list_name in ['weather_columns', 'static_columns']:
            col_list = getattr(args, col_list_name)
            for alias in ['doy', 'yday']:
                if alias in col_list:
                    col_list.remove(alias)

        # Ensure sin/cos are in weather_columns (time-varying)
        if 'doy_sin' not in args.weather_columns:
            args.weather_columns.extend(['doy_sin', 'doy_cos'])

    # ------------------------------------------------------------------
    # Now that feature lists are final, check that all columns exist
    # ------------------------------------------------------------------
    missing_columns = [
        col for col in (args.weather_columns + args.static_columns)
        if col not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    # ------------------------------------------------------------------
    # Categorical encoders for static columns (fit on train only)
    # ------------------------------------------------------------------
    categorical_encoders = {}
    categorical_decoders = {}

    for col in args.static_columns:
        if col in df_train.columns and df_train[col].dtype in ['object', 'string']:
            enc = LabelEncoder()
            # Fit on pre-2024 training data only
            df_train[col] = enc.fit_transform(df_train[col].astype(str))

            # Map 2024 data through encoder, unseen values -> -1
            if not df_2024.empty and col in df_2024.columns:
                df_2024[col] = df_2024[col].astype(str).map(
                    lambda x: enc.transform([x])[0] if x in enc.classes_ else -1
                )

            categorical_encoders[col] = enc
            categorical_decoders[col] = dict(
                zip(enc.transform(enc.classes_), enc.classes_)
            )

    # ------------------------------------------------------------------
    # Window creation helper
    # ------------------------------------------------------------------
    def create_windows(data):
        windows, labels, statics, years, groups = [], [], [], [], []

        # For grouping into individuals, ignore lat/long (they're engineered)
        grouping_cols = [c for c in args.static_columns
                         if c not in ['Site_lat', 'Site_long']]
        grouping_cols.extend(['Plot', 'Ind', 'Yrm', 'Tcode', 'gr'])
        grouping_cols = list(dict.fromkeys(grouping_cols))  # deduplicate

        relevant_cols = [
            c for c in (args.weather_columns + args.static_columns +
                        ['days_until_senescence'])
            if c in data.columns
        ]

        grouped = data.groupby(grouping_cols, sort=False)
        for key, g in grouped:
            g = g.sort_values('yday').reset_index(drop=True)

            for i in range(0, len(g) - args.time_interval + 1, args.stride):
                window = g.iloc[i:i + args.time_interval].copy()

                # Skip windows with any NaNs in relevant columns
                if window[relevant_cols].isna().any().any():
                    continue

                windows.append(window)
                labels.append(window['days_until_senescence'].iloc[-1])
                statics.append(
                    window.iloc[-1][args.static_columns].astype(float).values
                )
                years.append(int(window['Yrm'].iloc[-1]))
                groups.append("_".join(map(str, key)))

        return (
            windows,
            np.array(labels),
            np.array(statics),
            np.array(years),
            np.array(groups),
        )

    # Build windows for pre-2024 and for 2024 holdout
    train_windows, train_labels, train_static, train_years, train_groups = \
        create_windows(df_train)

    if not df_2024.empty:
        val_2024_windows, val_2024_labels, val_2024_static, val_2024_years, val_2024_groups = \
            create_windows(df_2024)
    else:
        val_2024_windows, val_2024_labels, val_2024_static, val_2024_years, val_2024_groups = \
            [], np.array([]), np.array([]), np.array([]), np.array([])

    return (
        (train_windows, train_labels, train_static, train_years, train_groups),
        (val_2024_windows, val_2024_labels, val_2024_static, val_2024_years, val_2024_groups),
        (categorical_encoders, categorical_decoders),
    )

def split_and_scale(X_weather, X_static, y, years, groups, holdout_year=2024):
    """Returns scaled train/val + raw holdout, plus fitted scalers and indices."""
    mask_pre = years < holdout_year
    Xw_pre, Xs_pre, y_pre, groups_pre = X_weather[mask_pre], X_static[mask_pre], y[mask_pre], groups[mask_pre]

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(Xw_pre, y_pre, groups=groups_pre))

    Xw_train, Xw_val = Xw_pre[train_idx], Xw_pre[val_idx]
    Xs_train, Xs_val = Xs_pre[train_idx], Xs_pre[val_idx]
    y_train,  y_val  = y_pre[train_idx],  y_pre[val_idx]

    weather_scaler = StandardScaler().fit(Xw_train.reshape(-1, Xw_train.shape[-1]))
    static_scaler  = StandardScaler().fit(Xs_train)
    target_scaler  = StandardScaler().fit(y_train.reshape(-1, 1))

    Xw_train_s = weather_scaler.transform(Xw_train.reshape(-1, Xw_train.shape[-1])).reshape(Xw_train.shape)
    Xw_val_s   = weather_scaler.transform(Xw_val.reshape(-1, Xw_val.shape[-1])).reshape(Xw_val.shape)
    Xs_train_s = static_scaler.transform(Xs_train)
    Xs_val_s   = static_scaler.transform(Xs_val)

    y_train_s  = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_s    = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    return (Xw_train_s, Xs_train_s, y_train_s, Xw_val_s, Xs_val_s, y_val_s,
            weather_scaler, static_scaler, target_scaler, train_idx, val_idx)



def build_model(weather_input_shape, static_input_shape):
    """
    New architecture:
      - Weather: 2x LSTM -> BN -> self-attention -> pooling (max + mean)
      - Static: small MLP, heavy dropout + L2, then scaled down
      - Fusion: concat(weather_repr, scaled_static) -> dense head
      - Output: main_head + weather-only residual head
    """

    # ─── Inputs ───────────────────────────────────────────────
    weather_input = Input(shape=weather_input_shape, name="weather_input")  # (B, T, Fw)
    static_input  = Input(shape=static_input_shape,  name="static_input")   # (B, Fs)

    # ─── Weather Path (LSTM → LSTM → BN → Self-Attn) ─────────
    x = LSTM(
        128,
        return_sequences=True,
        recurrent_dropout=0.15,
        kernel_regularizer=regularizers.l2(1e-4),
        name="lstm_1",
    )(weather_input)

    x = LSTM(
        64,
        return_sequences=True,
        recurrent_dropout=0.15,
        kernel_regularizer=regularizers.l2(1e-4),
        name="lstm_2",
    )(x)

    x = BatchNormalization(name="weather_bn")(x)

    # Self-attention over time (non-causal; we're using all T days)
    x_att = MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        name="self_attention",
    )(x, x, use_causal_mask=False)

    x = Add(name="attn_residual")([x, x_att])
    x = LayerNormalization(name="attn_norm")(x)

    # Pool over time: max + mean
    x_max  = GlobalMaxPooling1D(name="time_max")(x)          # (B, D)
    x_mean = GlobalAveragePooling1D(name="time_mean")(x)     # (B, D)
    weather_repr = Concatenate(name="weather_pool_concat")([x_max, x_mean])

    # ─── Static Path (small, heavily regularized) ────────────
    s = BatchNormalization(name="static_bn_in")(static_input)

    s = Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(5e-3),
        bias_regularizer=regularizers.l2(5e-3),
        name="static_dense1",
    )(s)
    s = BatchNormalization(name="static_bn1")(s)
    s = Dropout(0.7, name="static_drop1")(s)

    s = Dense(
        8,
        activation="relu",
        kernel_regularizer=regularizers.l2(5e-3),
        bias_regularizer=regularizers.l2(5e-3),
        name="static_dense2",
    )(s)
    s = BatchNormalization(name="static_bn2")(s)
    s = Dropout(0.7, name="static_drop2")(s)

    # Scale static contribution down so it behaves more like a small bias term
    s = Lambda(lambda z: 0.3 * z, name="scale_static")(s)

    # ─── Fusion & Dense Head ─────────────────────────────────
    fusion = Concatenate(name="fusion_concat")([weather_repr, s])

    z = Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="head_dense1",
    )(fusion)
    z = BatchNormalization(name="head_bn1")(z)
    z = Dropout(0.4, name="head_drop1")(z)

    z = Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="head_dense2",
    )(z)
    z = BatchNormalization(name="head_bn2")(z)
    z = Dropout(0.2, name="head_drop2")(z)

    main_out = Dense(1, name="main_out")(z)

    # ─── Weather-only Residual Head ──────────────────────────
    # This gives the network a direct linear readout from weather,
    # encouraging it to lean on weather features.
    aux_out = Dense(
        1,
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name="weather_residual",
    )(weather_repr)

    # Final prediction = main_head + weather-only residual
    final_out = Add(name="final_output")([main_out, aux_out])

    model = Model(
        inputs=[weather_input, static_input],
        outputs=final_out,
        name="senescence_model_v2",
    )
    return model



def save_model_diagram(model, out_path="plots/model_architecture.png"):
    """Save a PNG diagram of the model. Falls back to JSON+TXT if graphviz/pydot are missing."""
    import os, io, json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        from tensorflow.keras.utils import plot_model
        plot_model(
            model,
            to_file=out_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=True,
            dpi=200
        )
        print(f"[diagram] saved → {out_path}")
        return out_path
    except Exception as e:
        # Fallbacks: save JSON and a text summary so you still have artifacts
        json_path = out_path.replace(".png", ".json")
        txt_path  = out_path.replace(".png", ".txt")
        with open(json_path, "w") as f:
            f.write(json.dumps(json.loads(model.to_json()), indent=2))
        s = io.StringIO()
        model.summary(print_fn=lambda line: s.write(line + "\n"))
        with open(txt_path, "w") as f:
            f.write(s.getvalue())
        print(f"[diagram] Could not write PNG ({e}). Wrote {json_path} and {txt_path} instead.")
        return None

def train(model, X_weather, X_static, y, years, groups, holdout_year=2024):
    """
    Group-based split on pre-holdout years.
    Train/Val split comes from pre-holdout data.
    Test split is the holdout year if available, otherwise part of pre-holdout.
    """
    # --- Separate holdout year ---
    mask_pre = years < holdout_year
    mask_holdout = years == holdout_year

    Xw_pre, Xs_pre, y_pre, groups_pre = X_weather[mask_pre], X_static[mask_pre], y[mask_pre], groups[mask_pre]
    Xw_holdout, Xs_holdout, y_holdout = X_weather[mask_holdout], X_static[mask_holdout], y[mask_holdout]

    # --- Group split: train vs val ---
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(Xw_pre, y_pre, groups=groups_pre))

    Xw_train, Xw_val = Xw_pre[train_idx], Xw_pre[val_idx]
    Xs_train, Xs_val = Xs_pre[train_idx], Xs_pre[val_idx]
    y_train, y_val   = y_pre[train_idx], y_pre[val_idx]

    # --- Fit scalers on TRAIN only ---
    weather_scaler = StandardScaler().fit(Xw_train.reshape(-1, Xw_train.shape[-1]))
    static_scaler  = StandardScaler().fit(Xs_train)
    target_scaler  = StandardScaler().fit(y_train.reshape(-1, 1))

    # Transform X only
    Xw_train = weather_scaler.transform(Xw_train.reshape(-1, Xw_train.shape[-1])).reshape(Xw_train.shape)
    Xw_val   = weather_scaler.transform(Xw_val.reshape(-1, Xw_val.shape[-1])).reshape(Xw_val.shape)
    Xs_train = static_scaler.transform(Xs_train)
    Xs_val   = static_scaler.transform(Xs_val)

    if len(Xw_holdout) > 0:
        Xw_test = weather_scaler.transform(Xw_holdout.reshape(-1, Xw_holdout.shape[-1])).reshape(Xw_holdout.shape)
        Xs_test = static_scaler.transform(Xs_holdout)
        y_test  = y_holdout   # keep RAW here
    else:
        Xw_test, Xs_test, y_test = Xw_val, Xs_val, y_val  # fallback

    # --- Scale only y_train for fitting ---
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()

    # --- Train model ---
    print(f"[train] train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    ]

    history = model.fit(
        [Xw_train, Xs_train], y_train_scaled,
        validation_data=([Xw_val, Xs_val], target_scaler.transform(y_val.reshape(-1, 1)).flatten()),
        epochs=250, batch_size=2048, callbacks=callbacks, verbose=1
    )

    return history, (Xw_train, Xs_train, y_train, Xw_test, Xs_test, y_test), (weather_scaler, static_scaler, target_scaler)


def evaluate_model(history, model, data_splits, args,
                   val_2024_data=None, rolling_windows=None,
                   val_2024_windows=None, target_scaler=None,
                   weather_scaler=None, static_scaler=None,
                   plot_dir="plots"):
    """
    Evaluate model on train/test + optional 2024 holdout,
    with proper scaling/unscaling.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # 1) Loss curves
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss_plot.png'))
    plt.close()

    # 2) MAE curves
    plt.figure()
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Training vs. Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (days)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'mae_plot.png'))
    plt.close()

    # Unpack splits
    Xw_train, Xs_train, y_train_raw, Xw_test, Xs_test, y_test_raw = data_splits

    # Train preds
    y_train_scaled = target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
    y_train_pred = model.predict([Xw_train, Xs_train]).flatten()
    y_train_pred = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_train = y_train_raw  # already raw

    # Test preds
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    y_test_pred = model.predict([Xw_test, Xs_test]).flatten()
    y_test_pred = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    y_test = y_test_raw

    # Metrics
    def get_metrics(y_true, y_pred):
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        r2 = np.corrcoef(y_true, y_pred)[0,1]**2 if len(y_true) > 1 else np.nan
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        return r2, rmse

    train_r2, train_rmse = get_metrics(y_train, y_train_pred)
    test_r2, test_rmse = get_metrics(y_test, y_test_pred)

    # --- Scatter plot (train/test) ---
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.scatter(y_train, y_train_pred, alpha=0.5, c='blue',
                label=f'Training (R²={train_r2:.3f}, RMSE={train_rmse:.1f})')
    plt.scatter(y_test, y_test_pred, alpha=0.5, c='red',
                label=f'Test (R²={test_r2:.3f}, RMSE={test_rmse:.1f})')

    all_min = min(min(y_train), min(y_test))
    all_max = max(max(y_train), max(y_test))
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', label='Perfect Prediction')
    plt.xlabel('True Days to Senescence')
    plt.ylabel('Predicted Days to Senescence')
    plt.title('Model Predictions (Training/Test)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 2024 holdout ---
    plt.subplot(122)
    if val_2024_data and len(val_2024_data[0]) > 0:
        Xw_2024, Xs_2024, y_2024_raw = val_2024_data

        # ⬇️ NEW: scale with training scalers
        Xw_2024 = weather_scaler.transform(
            Xw_2024.reshape(-1, Xw_2024.shape[-1])
        ).reshape(Xw_2024.shape)
        Xs_2024 = static_scaler.transform(Xs_2024)

        y_2024_pred = model.predict([Xw_2024, Xs_2024]).flatten()
        y_2024_pred = target_scaler.inverse_transform(
            y_2024_pred.reshape(-1, 1)
        ).flatten()
        y_2024 = y_2024_raw

        val_r2, val_rmse = get_metrics(y_2024, y_2024_pred)

        plt.scatter(y_2024, y_2024_pred, alpha=0.5, c='green',
                    label=f'2024 (R²={val_r2:.3f}, RMSE={val_rmse:.1f})')
        val_min, val_max = min(y_2024), max(y_2024)
        plt.plot([val_min, val_max], [val_min, val_max], 'k--', label='Perfect Prediction')
        plt.xlabel('True Days to Senescence')
        plt.ylabel('Predicted Days to Senescence')
        plt.title('Model Predictions (2024)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No 2024 data available',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'prediction_scatter.png'))
    plt.close()

    # Residuals
    plot_residual_analysis(
    model,
    rolling_windows,
    args.weather_columns,
    target_scaler,
    args.static_columns,
    plot_dir=plot_dir,
    weather_scaler=weather_scaler,
    static_scaler=static_scaler,
    )
    print(f"[evaluate_model] Plots saved to '{plot_dir}/'")

def evaluate_site_accuracy(model, rolling_windows, y_true, data_columns,
                           target_scaler, static_columns, weather_scaler, static_scaler):
    X_weather = np.array([w[data_columns].values for w in rolling_windows])
    X_static  = np.array([w.iloc[-1][static_columns].astype(float).values for w in rolling_windows])

    # NEW: scale inputs
    X_weather = weather_scaler.transform(X_weather.reshape(-1, X_weather.shape[-1])).reshape(X_weather.shape)
    X_static  = static_scaler.transform(X_static)

    y_pred = model.predict([X_weather, X_static]).flatten()
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # DO NOT inverse_transform y_true
    records = []
    for window_df, t, p in zip(rolling_windows, y_true, y_pred):
        site = window_df['Site'].iloc[0]
        src  = window_df['Src'].iloc[0]
        records.append((site, src, t, p))

    results = pd.DataFrame(records, columns=['Site','Src','True','Pred'])
    site_mae = (
        results
        .groupby(['Site','Src'])
        .apply(lambda df: np.mean(np.abs(df['True'] - df['Pred'])))
        .reset_index(name='MAE')
    )

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    x_labels = [f"{row.Site}-{row.Src}" for row in site_mae.itertuples()]
    heights = site_mae['MAE']
    plt.bar(x_labels, heights)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean Absolute Error (days)")
    plt.title("Model MAE by Site-Src Combination")
    plt.tight_layout()
    plt.savefig('plots/site_mae.png')
    plt.close()

    return site_mae

def plot_site_mae(site_mae, categorical_decoders, plot_path='plots/site_mae.png'):
    """
    Plot site MAE with decoded site/src labels.
    """
    # Decode site/src if decoders are provided
    if 'Site' in categorical_decoders and 'Src' in categorical_decoders:
        site_mae['Site'] = site_mae['Site'].map(categorical_decoders['Site'])
        site_mae['Src'] = site_mae['Src'].map(categorical_decoders['Src'])
    x_labels = [f"{row.Site}-{row.Src}" for row in site_mae.itertuples()]
    heights = site_mae['MAE']
    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, heights)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean Absolute Error (days)")
    plt.title("Model MAE by Site-Src Combination")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def plot_residual_analysis(
    model,
    rolling_windows,
    weather_columns,
    target_scaler,
    static_columns,
    plot_dir="plots",
    weather_scaler=None,
    static_scaler=None
):


    # Build arrays
    X_weather = np.stack(
        [w[weather_columns].to_numpy(dtype=float) for w in rolling_windows], axis=0
    )  # shape: (n_samples, time, n_features)
    X_static = np.stack(
        [w.iloc[-1][static_columns].astype(float).to_numpy() for w in rolling_windows],
        axis=0
    )  # shape: (n_samples, n_static)
    y_true = np.array(
        [float(w["days_until_senescence"].iloc[-1]) for w in rolling_windows]
    )

    # Optional scaling
    if weather_scaler is not None:
        ws = X_weather.reshape(-1, X_weather.shape[-1])
        ws = weather_scaler.transform(ws)
        X_weather = ws.reshape(X_weather.shape)

    if static_scaler is not None:
        X_static = static_scaler.transform(X_static)

    # Predict and inverse-transform targets if requested
    y_pred = model.predict([X_weather, X_static], verbose=0).reshape(-1, 1)
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred)
    y_pred = y_pred.ravel()

    residuals = y_pred - y_true

    # Drop non-finite rows (protect downstream math)
    finite_mask = np.isfinite(residuals)
    if not np.all(finite_mask):
        X_weather = X_weather[finite_mask]
        X_static  = X_static[finite_mask]
        y_true    = y_true[finite_mask]
        residuals = residuals[finite_mask]

    n = len(weather_columns)
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = np.atleast_1d(axs).ravel()

    corrs = []

    for idx, feature in enumerate(weather_columns):
        ax = axs[idx]
        # Mean across time for each sample for this feature
        feature_means = np.nanmean(X_weather[:, :, idx], axis=1)

        # Keep only finite pairs
        m = np.isfinite(feature_means) & np.isfinite(residuals)
        x = feature_means[m]
        r = residuals[m]

        ax.scatter(x, r, alpha=0.5)
        ax.axhline(0, linestyle="--")

        # Trend line + Pearson r if well-defined
        if x.size >= 2 and np.std(x) > 0 and np.std(r) > 0:
            z = np.polyfit(x, r, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle="--", alpha=0.8)
            corr = float(np.corrcoef(x, r)[0, 1])
        else:
            corr = np.nan

        corrs.append((feature, corr))
        ax.set_xlabel(f"Mean {feature}")
        ax.set_ylabel("Residual (Predicted - Actual)")
        ax.set_title(f"Residuals vs {feature} | r={corr:.3f}")
        ax.grid(True, alpha=0.3)

    # Remove extra axes if grid > n
    for j in range(n, rows * cols):
        fig.delaxes(axs[j])

    os.makedirs(plot_dir, exist_ok=True)
    out_png = os.path.join(plot_dir, "residual_analysis.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Save correlations to CSV for quick triage
    corr_csv = os.path.join(plot_dir, "residual_feature_correlations.csv")
    with open(corr_csv, "w") as f:
        f.write("feature,pearson_r\n")
        for feat, r in corrs:
            if np.isnan(r):
                # leave blank if correlation couldn't be computed
                f.write(f"{feat},\n")
            else:
                f.write(f"{feat},{r:.6f}\n")

    return out_png, corr_csv

def main():
    args = parse_args()
    # Load and prepare data
    (train_windows, labels, train_static, years, groups), \
    (val_2024_windows, val_2024_labels, val_2024_static, val_2024_years, val_2024_groups), \
    (categorical_encoders, categorical_decoders) = load_data(args)

    # Convert training windows into arrays
    X_weather = np.array([w[args.weather_columns].values for w in train_windows])
    X_static = train_static
    y = labels

    weather_shape = (args.time_interval, len(args.weather_columns))
    static_shape = (len(args.static_columns),)

    # Build and train final model
    strategy = tf.distribute.MirroredStrategy()
    

    with strategy.scope():
        model = build_model(weather_shape, static_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss='huber',
            metrics=['mae']
        )
        save_model_diagram(model, out_path="plots/model_architecture.png")

    print(model.summary())
    history, splits, scalers = train(model, X_weather, X_static, y, years, groups, holdout_year=2024)
    weather_scaler, static_scaler, target_scaler = scalers

    artifacts = {
        "weather_columns": args.weather_columns,
        "static_columns": args.static_columns,
        "time_interval": args.time_interval,
        # label encoders from load_data()
        "categorical_encoders": {k: {"classes_": v.classes_.tolist()} for k, v in categorical_encoders.items()}
    }
    os.makedirs("models", exist_ok=True)
    dump({"weather_scaler": weather_scaler,
        "static_scaler": static_scaler,
        "target_scaler": target_scaler}, "models/scalers.joblib")
    with open("models/artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    print("Saved preprocess artifacts → models/scalers.joblib and models/artifacts.json")

    # Save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'senescence_model_lead.keras'))
    print(f"Saved model to {model_dir}/senescence_model_lead")

    # Evaluate on train/val + 2024 holdout
    evaluate_model(
        history, model, splits, args,
        val_2024_data=(np.array([w[args.weather_columns].values for w in val_2024_windows]),
                       val_2024_static, val_2024_labels) if len(val_2024_windows) > 0 else None,
        rolling_windows=train_windows,
        val_2024_windows=val_2024_windows,
        target_scaler=target_scaler,
        weather_scaler=weather_scaler,
        static_scaler=static_scaler
    )

    # Site-level accuracy
    site_acc_df = evaluate_site_accuracy(
        model, train_windows, labels,
        args.weather_columns, target_scaler, args.static_columns,
        weather_scaler, static_scaler
    )
    plot_site_mae(site_acc_df, categorical_decoders, plot_path='plots/site_mae.png')
    print("\n[main] Site-source accuracy:")
    print(site_acc_df.to_string(index=False))


if __name__ == '__main__':
    main()

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input data file.')
    parser.add_argument('--time_interval', type=int, default=7, help='Period of time from the data to train and predict')
    parser.add_argument('--subset_size', type=int, default=None, help='Number of rows to load for testing purposes.')
    parser.add_argument('--data_columns', type=str, nargs='+', required=True, 
                        help='List of column names to use as input features for the model.')
    
    return parser.parse_args()

def load_data(args):
    cache_file = f"{os.path.splitext(args.file_path)[0]}_rolling_windows.npz"
    
    # Check if the preprocessed file exists
    if os.path.exists(cache_file):
        print(f"Loading preprocessed data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        rolling_windows = data['rolling_windows']
        # Convert numpy array of objects back to a list of DataFrames
        rolling_windows = [pd.DataFrame(window) for window in rolling_windows]
        return rolling_windows
    
    df = pd.read_csv(args.file_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Load a subset of the data if subset_size is specified
    if args.subset_size:
        df = df.head(args.subset_size)
        print(f"Loaded a subset of the data with {args.subset_size} rows.")
    
    print(df.columns)
    
    # Ensure the target column is included
    if 'senescence' not in df.columns:
        raise ValueError("The target column 'senescence' is not present in the dataframe.")
    
    # Validate that all specified data_columns exist in the DataFrame
    missing_columns = [col for col in args.data_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"None of {missing_columns} are in the DataFrame columns: {df.columns.tolist()}")
    
    # Create rolling windows for each unique combination of "Site, Src, Plot, Ind, Year, and Tcode"
    unique_groups = df.groupby(['Site', 'Src', 'Plot', 'Ind', 'Yrm', 'Tcode'])
    time_interval = args.time_interval
    rolling_windows = []
    
    for _, group in unique_groups:
        group = group.reset_index(drop=True)
        for i in range(len(group) - time_interval + 1):
            window = group.iloc[i:i + time_interval]
            rolling_windows.append(window)
    
    # Save the rolling windows to a file
    print(f"Saving preprocessed data to {cache_file}")
    np.savez_compressed(cache_file, rolling_windows=np.array(rolling_windows, dtype=object))
    
    return rolling_windows

def build_model(input_shape):
    """
    Builds an LSTM model for predicting the last senescence boolean value in the time series.
    
    Args:
        input_shape (tuple): Shape of the input data (time_steps, num_features).
    
    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Predicting a boolean value
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train(model, data, time_interval, data_columns):
    """
    Splits the data into train, validation, and test sets, and trains the model.
    
    Args:
        model (tf.keras.Model): The LSTM model to train.
        data (list): List of rolling windows (dataframes).
        time_interval (int): Number of time steps in each rolling window.
        data_columns (list): List of column names to use as input features.
    
    Returns:
        tuple: History object, train/test data splits (X_train, y_train, X_test, y_test).
    """
    # Prepare input and output data
    X = np.array([window[data_columns].values for window in data])  # Selected columns as input features
    y = np.array([window['senescence'].iloc[-1] for window in data])  # Target column's last value
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32
    )

    return history, (X_train, y_train, X_test, y_test)

def evaluate_model(history, model, data_splits, plot_dir="plots"):
    """
    Evaluates the model's performance and saves plots of training/validation loss, accuracy, 
    and observed vs. predicted values for train and test sets.
    
    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit().
        model (tf.keras.Model): Trained model.
        data_splits (tuple): Tuple containing train/test data splits (X_train, y_train, X_test, y_test).
        plot_dir (str): Directory to save the plots.
    """
    # Create the plots directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot training and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss_plot.png'))
    plt.close()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'accuracy_plot.png'))
    plt.close()

    # Observed vs. Predicted plot for train and test sets
    X_train, y_train, X_test, y_test = data_splits
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    plt.figure()
    plt.scatter(y_train, y_train_pred, label='Train Set', alpha=0.6, color='blue')
    plt.scatter(y_test, y_test_pred, label='Test Set', alpha=0.6, color='orange')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Fit')  # Diagonal line for reference
    plt.title('Observed vs. Predicted')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'observed_vs_predicted.png'))
    plt.close()

    print(f"Plots saved to {plot_dir}")

def main():
    args = parse_args()

    data = load_data(args)

    # Determine input shape for the model
    time_interval = args.time_interval
    num_features = len(args.data_columns)  # Number of specified input features
    input_shape = (time_interval, num_features)

    # Build and train the model
    model = build_model(input_shape)
    history, data_splits = train(model, data, time_interval, args.data_columns)

    # Evaluate the model and save performance plots
    evaluate_model(history, model, data_splits)

if __name__ == "__main__":
    main()
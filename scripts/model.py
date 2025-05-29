import tensorflow as tf
import pandas as pd
import numpy as NotImplemented
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input data file.')
    parser.add_argument('--time_interval', type=int, default=7, help='Period of time from the data to train and predict')
    parser.add_argument('--data_types', type=+str, nargs='+', default=['dayl (s)','prcp (mm/day)','srad (W/m^2)','swe (kg/m^2)','tmax (deg c)','tmin (deg c)','vp (Pa)'], help='Types of data to use for training.')
    pass

def load_data():
    pass

def build_model():
    pass

def train(model):
    pass

def evaluate_model(model):
    pass


def main():
    args = parse_args()

    data = load_data(args.file_path)

    pass


if __name__ == "__main__":
    main()
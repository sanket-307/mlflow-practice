import argparse
from dataprocessing.temp import sum
from dataprocessing.preprocess import init_preprocess
from dataprocessing.ingest_data import fetch_housing_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing script to preprocess raw data"
    )
    parser.add_argument(
        "-a",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-b",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="data/raw",
        help="mention input_folder path to load your raw dataset",
    )
    parser.add_argument(
        "-f",
        "--input_filename",
        type=str,
        help="mention input_file name to load your raw dataset",
        default="housing.csv",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="mention output_folder path to store your preprocessed,fetaure engineered train and validation datasets",
        default="data/processed",
    )
    parser.add_argument(
        "-r",
        "--output_folder_rawdata",
        help="mention output_folder path where you want to store raw downloaded dataset",
        default="data/raw",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    inputs = parse_args()
    sum(inputs.a, inputs.b)
    fetch_housing_data(inputs.output_folder_rawdata)
    init_preprocess(inputs.input_folder, inputs.input_filename, inputs.output_folder)

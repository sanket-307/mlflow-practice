import os
import tarfile
from six.moves import urllib
import argparse
import logging

# from pathlib import path

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--output_folder",
    help="mention output_folder path where you want to store raw downloaded dataset",
    default="data/raw",
)
args = parser.parse_args()
OUTPUT_PATH = args.output_folder
# print("ouptut_path", OUTPUT_PATH)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
# print("Housing path:", HOUSING_PATH)
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# print("Housing url:", HOUSING_URL)


def fetch_housing_data(housing_url=HOUSING_URL, output_path=OUTPUT_PATH):
    logging.basicConfig(
        filename="mlhousing.log",
        encoding="utf-8",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )
    logging.info("Fetching Housing Data")
    logging.debug("Fetching data from: %s", HOUSING_URL)
    os.makedirs(output_path, exist_ok=True)
    tgz_path = os.path.join(output_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=output_path)
    housing_tgz.close()

    pyfile_directory = os.getcwd()
    # print("current working directory :", pyfile_directory)
    os.chdir(OUTPUT_PATH)
    # print("changed current working directory :", os.getcwd())
    logging.debug("Location of data stored: %s", os.path.abspath("housing.csv"))
    logging.info("Housing Fetch Data Completed")
    os.chdir(pyfile_directory)
    # print("again changed current working directory :", os.getcwd())


fetch_housing_data(output_path=OUTPUT_PATH)

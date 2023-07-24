import os
import tarfile
from six.moves import urllib

# from base_logger import logging

import logging

logging = logging

logging.basicConfig(
    filename="logs/mlhousing.log",
    encoding="utf-8",
    format="%(asctime)s:%(levelname)s:%(message)s",
    level=logging.DEBUG,
)


def fetch_housing_data(OUTPUT_PATH):
    """Fetch the data from the url and store data in output_path folder

    Args:
        OUTPUT_PATH (string, optional)

    Return : None.

    """

    output_path = OUTPUT_PATH
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    logging.info("Fetching Housing Data")
    logging.debug("Fetching data from: %s", HOUSING_URL)
    os.makedirs(output_path, exist_ok=True)
    tgz_path = os.path.join(output_path, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
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

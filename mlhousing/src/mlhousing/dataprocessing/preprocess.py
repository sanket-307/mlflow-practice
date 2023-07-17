import argparse
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
)
import logging

cw_dir = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_folder",
    help="mention input_folder path to load your raw dataset",
    default="data/raw",
)
parser.add_argument(
    "-f",
    "--input_filename",
    help="mention input_file name to load your raw dataset",
    default="housing.csv",
)
parser.add_argument(
    "-o",
    "--output_folder",
    help="mention output_folder path to store your preprocessed, fetaure engineered train and validation datasets",
    default="data/processed",
)
args = parser.parse_args()
INPUT_PATH = args.input_folder
INPUT_FILE = args.input_filename
OUTPUT_PATH = args.output_folder
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_housing_data(housing_path=INPUT_PATH, input_file=INPUT_FILE):
    logging.basicConfig(
        filename="mlhousing.log",
        encoding="utf-8",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )
    logging.info("Loading Data in DataFrame")
    csv_path = os.path.join(housing_path, input_file)
    print(csv_path)
    housing = pd.read_csv(csv_path)
    return housing


housing = load_housing_data(INPUT_PATH, INPUT_FILE)
print("type housing ", type(housing))
print("housing data shape ", housing.shape)
print(housing.dtypes)


housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

print("housing data shape after feature engineering", housing_prepared.shape)
print("housing label_data shape after feature engineering", housing_labels.shape)

file_name = INPUT_FILE.split(".")[0]
training_file = file_name + "_train.csv"
label_file = file_name + "_label.csv"

housing_prepared.to_csv(os.path.join(OUTPUT_PATH, training_file), index=False)
housing_labels.to_csv(os.path.join(OUTPUT_PATH, label_file), index=False)

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

print("housing data shape after feature engineering", X_test_prepared.shape)
print("housing label_data shape after feature engineering", X_test_cat.shape)

test_filename = file_name + "_testdata.csv"
testlabel_filename = file_name + "_testlabel.csv"

X_test_prepared.to_csv(os.path.join(OUTPUT_PATH, test_filename), index=False)
X_test_cat.to_csv(os.path.join(OUTPUT_PATH, testlabel_filename), index=False)

os.chdir(OUTPUT_PATH)

logging.debug(
    "Processed and feature engineered training data stored at: %s",
    os.path.abspath(training_file),
)
logging.debug("Training labels data stored at: %s", os.path.abspath(label_file))
logging.debug(
    "Processed and feature engineered test data stored at: %s",
    os.path.abspath(test_filename),
)
logging.debug("Test labels data stored at: %s", os.path.abspath(testlabel_filename))

os.chdir(cw_dir)

logging.info("Preprocessing and Feature Engineering step completed.")

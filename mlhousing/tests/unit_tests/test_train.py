from os import path
import os
import pandas as pd
from src.mlhousing.training.train import (
    linear_regression,
    decision_tree,
    rf_rs_regression,
    rf_grid_regression,
)


def data_generator(training_data_path):
    training_data = os.path.join(training_data_path, "housing_train.csv")
    training_label = os.path.join(training_data_path, "housing_label.csv")

    X_train = pd.read_csv(training_data)
    y_train = pd.read_csv(training_label)

    X_train = X_train[:1000]
    y_train = y_train[:1000]

    return X_train, y_train


def test_linear_regression_save(params):
    X_train, y_train = data_generator(params["OUTPUT_PATH"])

    artifacts = params["ARTIFACTS_PATH"]

    linear_regression(X_train, y_train, artifacts)

    linear_pickle = os.path.join(artifacts, "linearmodel.pkl")
    assert path.exists(linear_pickle)


def test_decision_tree_save(params):
    X_train, y_train = data_generator(params["OUTPUT_PATH"])

    artifacts = params["ARTIFACTS_PATH"]

    decision_tree(X_train, y_train, artifacts)

    dt_pickle = os.path.join(artifacts, "dtmodel.pkl")
    assert path.exists(dt_pickle)


# def test_rf_rs_regression_save(params):
#     X_train, y_train = data_generator(params["OUTPUT_PATH"])

#     artifacts = params["ARTIFACTS_PATH"]

#     rf_rs_regression(X_train, y_train, artifacts)
#     rf_rs_pickle = os.path.join(artifacts, "rf_rs_model.pkl")
#     assert path.exists(rf_rs_pickle)


# def test_rf_grid_regression_save(params):
#     X_train, y_train = data_generator(params["OUTPUT_PATH"])

#     artifacts = params["ARTIFACTS_PATH"]

#     rf_grid_regression(X_train, y_train, artifacts)
#     rf_grid_pickle = os.path.join(artifacts, "rf_grid_model.pkl")
#     assert path.exists(rf_grid_pickle)

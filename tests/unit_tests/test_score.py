import pandas as pd
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from mlhousing.score import linear_score, dt_score, rf_rs_score, rf_grid_score


def data_generator(testing_data_path):
    """data genrator function to load processed testing data and label to do unit test of score.

    Args:
        training_data_path (string, mandatory)


    Return : X_test, y_test (Dataframe).

    """

    test_data = os.path.join(testing_data_path, "housing_testdata.csv")
    test_label = os.path.join(testing_data_path, "housing_testlabel.csv")

    X_test = pd.read_csv(test_data)
    y_test = pd.read_csv(test_label)

    return X_test, y_test


def test_linear_regression_load_score(params, mocker):
    """unit testing of linear regression.

    Args:
        params (dictionary from conftest.py)


    Return : assert, None.

    """
    mlflow_mock = mocker.MagicMock()

    X_test, y_test = data_generator(params["OUTPUT_PATH"])

    artifacts = params["ARTIFACTS_PATH"]

    rmse = linear_score(X_test, y_test, artifacts, mlflow_mock)

    linear_pickle = os.path.join(artifacts, "linearmodel.pkl")
    lin_reg = pickle.load(open(linear_pickle, "rb"))
    assert isinstance(lin_reg, LinearRegression)
    assert isinstance(rmse["Linear rmse"], float)
    assert rmse["Linear rmse"] >= 0.0

    mlflow_mock.start_run.assert_called()
    mlflow_mock.log_metric.assert_any_call("mse", rmse["Linear mse"])
    mlflow_mock.log_metric.assert_any_call("rmse", rmse["Linear rmse"])
    mlflow_mock.log_metric.assert_any_call("mae", rmse["Linear mae"])


def test_decision_tree_load_score(params, mocker):
    """unit testing of DT regression.

    Args:
        params (dictionary from conftest.py)


    Return : assert, None.

    """
    mlflow_mock = mocker.MagicMock()

    X_test, y_test = data_generator(params["OUTPUT_PATH"])

    artifacts = params["ARTIFACTS_PATH"]

    rmse = dt_score(X_test, y_test, artifacts, mlflow_mock)

    dt_pickle = os.path.join(artifacts, "dtmodel.pkl")
    tree_reg = pickle.load(open(dt_pickle, "rb"))
    assert isinstance(tree_reg, DecisionTreeRegressor)
    assert isinstance(rmse["DT RMSE"], float)
    assert rmse["DT RMSE"] >= 0.0

    mlflow_mock.start_run.assert_called()
    mlflow_mock.log_metric.assert_any_call("mse", rmse["DT MSE"])
    mlflow_mock.log_metric.assert_any_call("rmse", rmse["DT RMSE"])


def test_rf_rs_regression_load_score(params, mocker):
    """unit testing of random forest random search hyperparameter regression.

    Args:
        params (dictionary from conftest.py)


    Return : assert, None.

    """

    mlflow_mock = mocker.MagicMock()

    X_test, y_test = data_generator(params["OUTPUT_PATH"])

    artifacts = params["ARTIFACTS_PATH"]

    rmse = rf_rs_score(X_test, y_test, artifacts, mlflow_mock)

    rf_rs_pickle = os.path.join(artifacts, "rf_rs_model.pkl")
    rf_rs_model = pickle.load(open(rf_rs_pickle, "rb"))
    assert isinstance(rf_rs_model, RandomForestRegressor)
    assert isinstance(rmse["RT RS RMSE"], float)
    assert rmse["RT RS RMSE"] >= 0.0

    mlflow_mock.start_run.assert_called()
    mlflow_mock.log_metric.assert_any_call("mse", rmse["RT RS MSE"])
    mlflow_mock.log_metric.assert_any_call("rmse", rmse["RT RS RMSE"])


def test_rf_grid_regression_load_score(params, mocker):
    """unit testing of random forest grid search hyperparameter regression.

    Args:
        params (dictionary from conftest.py)


    Return : assert, None.

    """
    mlflow_mock = mocker.MagicMock()

    X_test, y_test = data_generator(params["OUTPUT_PATH"])

    artifacts = params["ARTIFACTS_PATH"]

    rmse = rf_grid_score(X_test, y_test, artifacts, mlflow_mock)

    rf_grid_pickle = os.path.join(artifacts, "rf_rs_model.pkl")
    rf_grid_model = pickle.load(open(rf_grid_pickle, "rb"))
    assert isinstance(rf_grid_model, RandomForestRegressor)
    assert isinstance(rmse["RT GRID RMSE"], float)
    assert rmse["RT GRID RMSE"] >= 0.0

    mlflow_mock.start_run.assert_called()
    mlflow_mock.log_metric.assert_any_call("mse", rmse["RT GRID MSE"])
    mlflow_mock.log_metric.assert_any_call("rmse", rmse["RT GRID RMSE"])

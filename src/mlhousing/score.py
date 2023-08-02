from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
import pickle

# from base_logger import logging

import logging

logging = logging

logging.basicConfig(
    filename="logs/mlhousing.log",
    encoding="utf-8",
    format="%(asctime)s:%(levelname)s:%(message)s",
    level=logging.DEBUG,
)


def init_score(testing_data_path, artifacts_path, mlflow):
    """function take commandline arguments from process.py and call different function to execute score operations for linear regression,
    decision tree, random forest random saerch and random forest grid search.

        Args:
            testing_data_path (string, mandatory)
            artifacts_path (string, mandatory)


        Return : None.

    """

    test_data = os.path.join(testing_data_path, "housing_testdata.csv")
    test_label = os.path.join(testing_data_path, "housing_testlabel.csv")

    X_test = pd.read_csv(test_data)
    y_test = pd.read_csv(test_label)

    print(
        "X_test data shape is: {} and y_test data shape is: {}".format(
            X_test.shape, y_test.shape
        )
    )

    linear_score(X_test, y_test, artifacts_path, mlflow)
    dt_score(X_test, y_test, artifacts_path, mlflow)
    rf_rs_score(X_test, y_test, artifacts_path, mlflow)
    rf_grid_score(X_test, y_test, artifacts_path, mlflow)


def linear_score(X_test, y_test, artifacts_path, mlflow):
    """load linearmodel.pkl from artifacts_path and perform prediction on
    X_test and calculate the RMSE score.

    Args:
        X_test (dataframe, mandatory)
        y_test (dataframe, mandatory)
        artifacts_path (string, mandatory)


    Return : dictionary {"Linear RMSE": lin_mae}.

    """
    with mlflow.start_run(run_name="score_linearregression", nested=True):
        linear_pickle = os.path.join(artifacts_path, "linearmodel.pkl")
        lin_reg = pickle.load(open(linear_pickle, "rb"))

        y_pred = lin_reg.predict(X_test)

        lin_mse = mean_squared_error(y_test, y_pred)
        lin_rmse = np.sqrt(lin_mse)
        print("Linear Regression RMSE:", lin_rmse)

        lin_mae = mean_absolute_error(y_test, y_pred)

        print("Linear Regression MAE:", lin_mae)

        logging.info("Linear Regression RMSE: %s", lin_rmse)
        logging.info("Linear Regression MAE: %s", lin_mae)

        mlflow.log_metric("mse", lin_mse)
        mlflow.log_metric("rmse", lin_rmse)
        mlflow.log_metric("mae", lin_mae)

        return {"Linear mse": lin_mse, "Linear rmse": lin_rmse, "Linear mae": lin_mae}


def dt_score(X_test, y_test, artifacts_path, mlflow):
    """load dtmodel.pkl from artifacts_path and perform prediction on
    X_test and calculate the RMSE score.

    Args:
        X_test (dataframe, mandatory)
        y_test (dataframe, mandatory)
        artifacts_path (string, mandatory)


    Return : dictionary {"DT RMSE": tree_rmse}.

    """
    with mlflow.start_run(run_name="score_DT", nested=True):
        dt_pickle = os.path.join(artifacts_path, "dtmodel.pkl")
        tree_reg = pickle.load(open(dt_pickle, "rb"))
        y_pred = tree_reg.predict(X_test)

        tree_mse = mean_squared_error(y_test, y_pred)
        tree_rmse = np.sqrt(tree_mse)
        tree_rmse
        print("DT RMSE:", tree_rmse)
        logging.info("DT RMSE: %s", tree_rmse)

        mlflow.log_metric("mse", tree_mse)
        mlflow.log_metric("rmse", tree_rmse)

        return {"DT RMSE": tree_rmse, "DT MSE": tree_mse}


def rf_rs_score(X_test, y_test, artifacts_path, mlflow):
    """load rf_rs_model.pkl from artifacts_path and perform prediction on
    X_test and calculate the RMSE score.

    Args:
        X_test (dataframe, mandatory)
        y_test (dataframe, mandatory)
        artifacts_path (string, mandatory)


    Return : dictionary {"RT RS RMSE": final_rmse}.

    """
    with mlflow.start_run(run_name="score_rf_rs", nested=True):
        rf_rs_pickle = os.path.join(artifacts_path, "rf_rs_model.pkl")
        rf_rs_model = pickle.load(open(rf_rs_pickle, "rb"))
        final_predictions = rf_rs_model.predict(X_test)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        print("Random forest Random search RMSE:", final_rmse)
        logging.info("Random forest Random search RMSE: %s", final_rmse)

        mlflow.log_metric("mse", final_mse)
        mlflow.log_metric("rmse", final_rmse)

        return {"RT RS RMSE": final_rmse, "RT RS MSE": final_mse}


def rf_grid_score(X_test, y_test, artifacts_path, mlflow):
    """load rf_grid_model.pkl from artifacts_path and perform prediction on
    X_test and calculate the RMSE score.

    Args:
        X_test (dataframe, mandatory)
        y_test (dataframe, mandatory)
        artifacts_path (string, mandatory)


    Return : dictionary {"RT GRID RMSE": final_rmse}.

    """
    with mlflow.start_run(run_name="score_rf_grid", nested=True):
        rf_grid_pickle = os.path.join(artifacts_path, "rf_grid_model.pkl")
        rf_grid_model = pickle.load(open(rf_grid_pickle, "rb"))
        final_predictions = rf_grid_model.predict(X_test)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)

        print("Random forest grid search RMSE:", final_rmse)
        logging.info("Random forest grid search RMSE: %s", final_rmse)

        mlflow.log_metric("mse", final_mse)
        mlflow.log_metric("rmse", final_rmse)

        return {"RT GRID RMSE": final_rmse, "RT GRID MSE": final_mse}

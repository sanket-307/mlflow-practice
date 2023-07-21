from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
import pickle
import logging


def init_score(testing_data_path, artifacts_path):
    logging.basicConfig(
        filename="mlhousing.log",
        encoding="utf-8",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )

    test_data = os.path.join(testing_data_path, "housing_testdata.csv")
    test_label = os.path.join(testing_data_path, "housing_testlabel.csv")

    X_test = pd.read_csv(test_data)
    y_test = pd.read_csv(test_label)

    print(
        "X_test data shape is: {} and y_test data shape is: {}".format(
            X_test.shape, y_test.shape
        )
    )

    linear_score(X_test, y_test, artifacts_path)
    dt_score(X_test, y_test, artifacts_path)
    rf_rs_score(X_test, y_test, artifacts_path)
    rf_grid_score(X_test, y_test, artifacts_path)


def linear_score(X_test, y_test, artifacts_path):
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

    return {"Linear RMSE": lin_mae}


def dt_score(X_test, y_test, artifacts_path):
    dt_pickle = os.path.join(artifacts_path, "dtmodel.pkl")
    tree_reg = pickle.load(open(dt_pickle, "rb"))
    y_pred = tree_reg.predict(X_test)

    tree_mse = mean_squared_error(y_test, y_pred)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse
    print("DT RMSE:", tree_rmse)
    logging.info("DT RMSE: %s", tree_rmse)
    return {"DT RMSE": tree_rmse}


def rf_rs_score(X_test, y_test, artifacts_path):
    rf_rs_pickle = os.path.join(artifacts_path, "rf_rs_model.pkl")
    rf_rs_model = pickle.load(open(rf_rs_pickle, "rb"))
    final_predictions = rf_rs_model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Random forest Random search RMSE:", final_rmse)
    logging.info("Random forest Random search RMSE: %s", final_rmse)
    return {"RT RS RMSE": final_rmse}


def rf_grid_score(X_test, y_test, artifacts_path):
    rf_grid_pickle = os.path.join(artifacts_path, "rf_grid_model.pkl")
    rf_grid_model = pickle.load(open(rf_grid_pickle, "rb"))
    final_predictions = rf_grid_model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Random forest grid search RMSE:", final_rmse)
    logging.info("Random forest grid search RMSE: %s", final_rmse)
    return {"RT GRID RMSE": final_rmse}

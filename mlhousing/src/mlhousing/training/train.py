from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import os
import pickle
from scipy.stats import randint
import logging


def init_training(training_data_path, artifacts_path):
    logging.basicConfig(
        filename="mlhousing.log",
        encoding="utf-8",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )

    training_data = os.path.join(training_data_path, "housing_train.csv")
    training_label = os.path.join(training_data_path, "housing_label.csv")

    X_train = pd.read_csv(training_data)
    y_train = pd.read_csv(training_label)

    print(
        "X_tarin data shape is: {} and y_train data shape is: {}".format(
            X_train.shape, y_train.shape
        )
    )

    linear_regression(X_train, y_train, artifacts_path)
    decision_tree(X_train, y_train, artifacts_path)
    rf_rs_regression(X_train, y_train, artifacts_path)
    rf_grid_regression(X_train, y_train, artifacts_path)


def linear_regression(X_train, y_train, artifacts_path):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    linear_pickle = os.path.join(artifacts_path, "linearmodel.pkl")
    pickle.dump(lin_reg, open(linear_pickle, "wb"))
    logging.info("Linear Regression model trained")
    logging.debug("Linear Regression artifacts stored at: %s", linear_pickle)


def decision_tree(X_train, y_train, artifacts_path):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)

    dt_pickle = os.path.join(artifacts_path, "dtmodel.pkl")
    pickle.dump(tree_reg, open(dt_pickle, "wb"))
    logging.info("DT model trained")
    logging.debug("DT artifacts stored at: %s", dt_pickle)


def rf_rs_regression(X_train, y_train, artifacts_path):
    y_train = y_train.values.ravel()

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X_train, y_train)

    # cvres = rnd_search.cv_results_

    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(np.sqrt(-mean_score), params)

    feature_importances = rnd_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, X_train.columns), reverse=True)

    rf_rs_model = rnd_search.best_estimator_

    rf_rs_pickle = os.path.join(artifacts_path, "rf_rs_model.pkl")
    pickle.dump(rf_rs_model, open(rf_rs_pickle, "wb"))

    logging.info(
        "Random forest model trained using best hyperparameters of random search technique"
    )
    logging.debug(
        "Random forest by best hyperparameter of random search artifacts stored at: %s",
        rf_rs_pickle,
    )


def rf_grid_regression(X_train, y_train, artifacts_path):
    y_train = y_train.values.ravel()

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    grid_search.best_params_
    # cvres = grid_search.cv_results_

    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, X_train.columns), reverse=True)

    rf_grid_model = grid_search.best_estimator_

    rf_grid_pickle = os.path.join(artifacts_path, "rf_grid_model.pkl")
    pickle.dump(rf_grid_model, open(rf_grid_pickle, "wb"))

    logging.info(
        "Random forest model trained using best hyperparameters of grid search technique"
    )
    logging.debug(
        "Random forest by best hyperparameter of grid search artifacts stored at: %s",
        rf_grid_pickle,
    )

# from os import path
# import os
# import pandas as pd
# from mlhousing.train import (
#     linear_regression,
#     decision_tree,
# )


# def data_generator(training_data_path):
#     """data genrator function to load processed training data and label to do integration test of training.

#     Args:
#         training_data_path (string, mandatory)


#     Return : X_train, y_train (Dataframe).

#     """

#     training_data = os.path.join(training_data_path, "housing_train.csv")
#     training_label = os.path.join(training_data_path, "housing_label.csv")

#     X_train = pd.read_csv(training_data)
#     y_train = pd.read_csv(training_label)

#     X_train = X_train[:1000]
#     y_train = y_train[:1000]

#     return X_train, y_train


# def test_linear_regression_train_save(params, mocker):
#     """integration testing of linear regression.

#     Args:
#         params (dictionary from conftest.py)


#     Return : assert, None.

#     """

#     mlflow_mock = mocker.MagicMock()

#     X_train, y_train = data_generator(params["OUTPUT_PATH"])

#     artifacts = params["ARTIFACTS_PATH"]

#     linear_regression(X_train, y_train, artifacts, mlflow_mock)

#     linear_pickle = os.path.join(artifacts, "linearmodel.pkl")
#     assert path.exists(linear_pickle)

#     mlflow_mock.log_artifact.return_value = None

#     mlflow_mock.log_artifact.assert_called_with(linear_pickle)


# def test_decision_tree_train_save(params):
#     """integration testing of DT regression.

#     Args:
#         params (dictionary from conftest.py)


#     Return : assert, None.

#     """

#     X_train, y_train = data_generator(params["OUTPUT_PATH"])

#     artifacts = params["ARTIFACTS_PATH"]

#     decision_tree(X_train, y_train, artifacts)

#     dt_pickle = os.path.join(artifacts, "dtmodel.pkl")
#     assert path.exists(dt_pickle)

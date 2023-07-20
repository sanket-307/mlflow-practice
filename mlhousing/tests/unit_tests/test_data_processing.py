# import sys
import pandas as pd
from sklearn.impute import SimpleImputer

from src.mlhousing.dataprocessing.preprocess import (
    load_housing_data,
    binning,
    startified_split,
    feature_engineering_traindataset,
    feature_engineering_testdataset,
)
from src.mlhousing.dataprocessing.temp import sum


def test_sum(params):
    sum(params["A"], params["B"])
    assert sum(params["A"], params["B"]) == params["C"]


def test_housing_df(params):

    df = load_housing_data(params["INPUT_PATH"], params["INPUT_FILE"])
    assert isinstance(df, pd.DataFrame)


def test_columns_present(get_rawdf):
    # ensures that the expected columns are all present
    assert "longitude" in get_rawdf.columns
    assert "latitude" in get_rawdf.columns
    assert "housing_median_age" in get_rawdf.columns
    assert "total_rooms" in get_rawdf.columns
    assert "population" in get_rawdf.columns
    assert "households" in get_rawdf.columns
    assert "median_income" in get_rawdf.columns
    assert "median_house_value" in get_rawdf.columns
    assert "ocean_proximity" in get_rawdf.columns


def test_non_empty(get_rawdf):
    # ensures that there is more than one row of data
    assert len(get_rawdf.index) != 0


def test_binning(get_rawdf):
    binning_df = binning(get_rawdf)
    assert "income_cat" in get_rawdf.columns
    assert isinstance(binning_df, pd.DataFrame)
    # return binning_df


def test_startified_split(get_binningdf):
    strat_train_set, strat_test_set = startified_split(get_binningdf)
    assert isinstance(strat_train_set, pd.DataFrame)
    assert isinstance(strat_test_set, pd.DataFrame)


def test_feature_engineering_traindataset(get_startified_split):
    df1, df2 = get_startified_split
    X_train, y_train, imputer = feature_engineering_traindataset(df1)
    assert "rooms_per_household" in X_train.columns
    assert "bedrooms_per_room" in X_train.columns
    assert "population_per_household" in X_train.columns
    assert "ocean_proximity_INLAND" in X_train.columns
    assert "ocean_proximity_ISLAND" in X_train.columns
    assert "ocean_proximity_NEAR BAY" in X_train.columns
    assert "ocean_proximity_NEAR OCEAN" in X_train.columns

    assert "ocean_proximity" not in X_train.columns
    assert "median_house_value" not in X_train.columns

    assert "median_house_value" in y_train.name
    # assert "median_house_value1" in y_train.name

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(imputer, SimpleImputer)


def test_feature_engineering_testdataset(
    get_startified_split, get_feature_engineering_traindataset
):
    df1, df2 = get_startified_split
    df11, df21, imputer = get_feature_engineering_traindataset

    X_test, y_test = feature_engineering_testdataset(df2, imputer)

    assert "rooms_per_household" in X_test.columns
    assert "bedrooms_per_room" in X_test.columns
    assert "population_per_household" in X_test.columns
    assert "ocean_proximity_INLAND" in X_test.columns
    assert "ocean_proximity_ISLAND" in X_test.columns
    assert "ocean_proximity_NEAR BAY" in X_test.columns
    assert "ocean_proximity_NEAR OCEAN" in X_test.columns

    assert "ocean_proximity" not in X_test.columns
    assert "median_house_value" not in X_test.columns

    # assert "median_house_value" in y_test.name

    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

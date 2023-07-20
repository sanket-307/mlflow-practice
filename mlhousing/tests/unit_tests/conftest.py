import pytest

from src.mlhousing.dataprocessing.preprocess import (
    load_housing_data,
    binning,
    startified_split,
    feature_engineering_traindataset,
)


def pytest_addoption(parser):
    parser.addoption("--i", type=str, default="data/raw")
    parser.addoption("--f", type=str, default="housing.csv")
    parser.addoption("--o", type=str, default="data/processed")
    parser.addoption("--a", type=int, default=1)
    parser.addoption("--b", type=int, default=2)
    parser.addoption("--c", type=int, default=3)


@pytest.fixture
def params(request):
    params = {}
    params["INPUT_PATH"] = request.config.getoption("--i")
    params["INPUT_FILE"] = request.config.getoption("--f")
    params["OUTPUT_PATH"] = request.config.getoption("--o")
    params["A"] = request.config.getoption("--a")
    params["B"] = request.config.getoption("--b")
    params["C"] = request.config.getoption("--c")

    return params


@pytest.fixture
def get_rawdf(params):
    df = load_housing_data(params["INPUT_PATH"], params["INPUT_FILE"])
    return df


@pytest.fixture
def get_binningdf(get_rawdf):
    df = binning(get_rawdf)
    return df


@pytest.fixture
def get_startified_split(get_binningdf):
    df = startified_split(get_binningdf)
    return df


@pytest.fixture
def get_feature_engineering_traindataset(get_startified_split):
    train_df, test_df = get_startified_split
    df = feature_engineering_traindataset(train_df)
    return df

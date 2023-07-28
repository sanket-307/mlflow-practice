import pytest

from mlhousing.preprocess import (
    load_housing_data,
    binning,
    startified_split,
    feature_engineering_traindataset,
)


def pytest_addoption(parser):
    """function to take command line arguments in pytest to test various functions.

    Args:
        parser


    Return : None.

    """

    # parser.addoption(
    #     "--i", help="raw dataset folder path", type=str, default="mlhousing/data/raw"
    # )
    # parser.addoption("--f", help="raw csv file name", type=str, default="housing.csv")
    # parser.addoption(
    #     "--o",
    #     help="processed training and testing data folder path",
    #     type=str,
    #     default="mlhousing/data/processed",
    # )
    # parser.addoption(
    #     "--m", help="artifact folder path", type=str, default="mlhousing/artifacts"
    # )

    parser.addoption(
        "--i", help="raw dataset folder path", type=str, default="data/raw"
    )
    parser.addoption("--f", help="raw csv file name", type=str, default="housing.csv")
    parser.addoption(
        "--o",
        help="processed training and testing data folder path",
        type=str,
        default="data/processed",
    )
    parser.addoption("--m", help="artifact folder path", type=str, default="artifacts")


@pytest.fixture
def params(request):
    """fixture function to access params in subsequent tests when and where required.

    Args:
        request


    Return : params.

    """

    params = {}
    params["INPUT_PATH"] = request.config.getoption("--i")
    params["INPUT_FILE"] = request.config.getoption("--f")
    params["OUTPUT_PATH"] = request.config.getoption("--o")
    params["ARTIFACTS_PATH"] = request.config.getoption("--m")

    return params


@pytest.fixture
def get_rawdf(params):
    """fixture function to raw data frame.

    Args:
        params


    Return : df (dataframe)

    """

    df = load_housing_data(params["INPUT_PATH"], params["INPUT_FILE"])
    return df


@pytest.fixture
def get_binningdf(get_rawdf):
    """fixture function to perform binning operation on raw data frame.

    Args:
        get_rawdf


    Return : df (dataframe)

    """

    df = binning(get_rawdf)
    return df


@pytest.fixture
def get_startified_split(get_binningdf):
    """fixture function to perform startified split operation on binning dataframe.

    Args:
        get_binningdf


    Return : df (tuple- train_df, test_df)

    """

    df = startified_split(get_binningdf)
    return df


@pytest.fixture
def get_feature_engineering_traindataset(get_startified_split):
    """fixture function to get feature engineeried dataset tuple (X_train, y_train) from get_startified_split returned tuple first element.

    Args:
        get_startified_split


    Return : df (tuple- X_train, y_train)

    """
    train_df, test_df = get_startified_split
    df = feature_engineering_traindataset(train_df)
    return df

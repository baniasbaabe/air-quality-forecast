import pandas as pd
import polars as pl
import pytest
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore

from src.feature_pipeline.load import Loader


@pytest.fixture
def loader():
    return Loader()


def test_get_project(mocker, loader):
    mocked_hopsworks_login = mocker.patch("src.utils.hopsworks_login")
    loader._get_project()
    mocked_hopsworks_login.assert_called_once()


def test_get_feature_store(mocker, loader):
    mocked_project = mocker.Mock()
    mocked_hopsworks_get_feature_store = mocker.patch(
        "src.utils.hopsworks_get_feature_store"
    )
    loader._get_feature_store(mocked_project)
    mocked_hopsworks_get_feature_store.assert_called_once_with(mocked_project)


def test_clean_feature_group(mocker, loader):
    mocked_feature_store = mocker.Mock(spec=FeatureStore)
    mocked_hopsworks_get_feature_group = mocker.patch(
        "src.utils.hopsworks_get_feature_group"
    )
    mocked_feature_group = mocked_hopsworks_get_feature_group.return_value
    loader._clean_feature_group(mocked_feature_store)
    mocked_feature_group.delete.assert_called_once()


def test_get_feature_group(mocker, loader):
    mocked_feature_store = mocker.Mock(spec=FeatureStore)
    mocked_hopsworks_get_feature_group = mocker.patch(
        "src.utils.hopsworks_get_feature_group"
    )
    loader._get_feature_group(mocked_feature_store)
    assert mocked_hopsworks_get_feature_group.call_count == 2


def test_insert_data_polars_dataframe(mocker, loader):
    mocked_feature_group = mocker.Mock(spec=FeatureGroup)
    mocked_df = mocker.Mock(spec=pl.DataFrame)
    mocker.patch(
        "src.utils.hopsworks_get_feature_group", return_value=mocked_feature_group
    )

    loader._insert_data(mocked_feature_group, mocked_df)

    # Assert that the to_pandas() method is called on the Polars DataFrame
    mocked_df.to_pandas.assert_called_once()

    # Assert that the insert method is called on the FeatureGroup
    mocked_feature_group.insert.assert_called_once_with(
        mocked_df.to_pandas(), wait=True
    )


def test_insert_data_other_dataframe(mocker, loader):
    mocked_feature_group = mocker.Mock(spec=FeatureGroup)

    # Mocking a DataFrame with to_pandas method
    mocked_df = mocker.Mock(spec=pd.DataFrame)
    mocked_df.to_pandas = mocker.Mock()  # Add the to_pandas attribute manually

    mocker.patch(
        "src.utils.hopsworks_get_feature_group", return_value=mocked_feature_group
    )

    loader._insert_data(mocked_feature_group, mocked_df)

    # Assert that the to_pandas() method is not called on non-Polars DataFrame
    mocked_df.to_pandas.assert_not_called()

    # Assert that the insert method is called on the FeatureGroup
    mocked_feature_group.insert.assert_called_once()

    # Assert that the insert method is called with the original DataFrame
    mocked_feature_group.insert.assert_called_once_with(mocked_df, wait=True)

import os

import pytest
from comet_ml import API

from src.inference_pipeline.model_loader import ModelLoader


@pytest.fixture
def loader():
    return ModelLoader(model_name="mockmodel")


@pytest.fixture
def comet_api_setup(mocker):
    mocker.patch.dict(
        os.environ,
        {"COMET_API_KEY": "mock_api_key", "COMET_WORKSPACE": "mock_workspace"},
    )

    mocked_api = mocker.patch.object(API, "get_latest_registry_model_version_details")
    mocked_api.return_value = {"versions": [{"version": "1"}]}

    return mocked_api


def test_get_current_model_version(loader, comet_api_setup):
    result = loader._get_current_model_version()

    # Assertions
    comet_api_setup.assert_called_once_with("mock_workspace", "mockmodel")
    assert result == "1"


def test_download_current_model(mocker, loader, comet_api_setup):
    mocked_download = mocker.patch.object(API, "download_registry_model")
    mocked_download.return_value = "mock_model_data"

    _ = loader._download_current_model()

    comet_api_setup.assert_called_once_with("mock_workspace", "mockmodel")
    mocked_download.assert_called_once_with(
        "mock_workspace",
        registry_name="mockmodel",
        version="1",
        output_path="./",
        expand=True,
    )

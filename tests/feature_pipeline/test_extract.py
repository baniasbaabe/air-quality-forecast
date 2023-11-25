from src.feature_pipeline.extract import SensorAPIClient


def test_fetch_data(mocker):
    mock_get = mocker.patch("httpx.get")

    return_value = {
        "SensorID": 1,
        "Timestamp": "2021-12-01T00:00:00Z",
        "PM10": 10,
        "PM2.5": 5,
    }

    mock_get.return_value.json.return_value = return_value

    client = SensorAPIClient("http://test.url")

    result = client.fetch_data()

    mock_get.assert_called_once_with("http://test.url", params=None, timeout=None)
    assert result == return_value

from abc import ABC, abstractmethod
from typing import Optional

import httpx
from loguru import logger


class APIClient(ABC):
    """Abstract Base Class for API Clients to fetch data.

    Args:
        ABC (_type_): Abstract Base Class
    """

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    @abstractmethod
    def fetch_data(self, params: Optional[dict] = None) -> dict:
        """Abstract method for fetching data from a data source.

        Args:
            params (Optional[dict], optional): Optional Parameters for requesting data.
            Defaults to None.

        Returns:
            dict: Returned Data.
        """
        pass


class SensorAPIClient(APIClient):
    """
    Concrete Implementation for the sensordata API.
    https://feinstaub.citysensor.de/


    Args:
        APIClient (_type_): _description_
    """

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    def fetch_data(self, params: Optional[dict] = None) -> dict:
        """Fetch Data from Sensor API and return it as a dictionary.

        Args:
            params (Optional[dict], optional): Optional Parameters like specific
            Sensor ID. Defaults to None.

        Returns:
            dict: Dictionary with the response from the
            API (SensorID, Timestamp, PM10, PM2.5)
        """
        logger.info("Fetching data from Sensor API...")
        response = httpx.get(self.api_url, params=params, timeout=None)
        return response.json()

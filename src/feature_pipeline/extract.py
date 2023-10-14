import httpx
from loguru import logger
from typing import Optional
from abc import ABC, abstractmethod


class APIClient(ABC):
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    @abstractmethod
    def fetch_data(self, params: Optional[dict] = None) -> dict:
        pass


class SensorAPIClient(APIClient):
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    def fetch_data(self, params: Optional[dict] = None) -> dict:
        logger.info("Fetching data from Sensor API...")
        response = httpx.get(self.api_url, params=params, timeout=None)
        return response.json()
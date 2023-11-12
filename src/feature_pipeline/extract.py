from typing import Optional

import httpx
import tenacity
from loguru import logger


class SensorAPIClient:
    """Concrete Implementation for the sensordata API.

    https://feinstaub.citysensor.de/
    """

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_fixed(10))
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

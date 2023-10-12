from extraction import SensorAPIClient

api = SensorAPIClient(api_url="https://feinstaub.citysensor.de/api/getdata")

print(api.fetch_data(params={"sensorid": "2446", "avg": "1", "span": "720"}))

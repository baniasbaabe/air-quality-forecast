from extract import SensorAPIClient


def main():
    sensor_api = SensorAPIClient(api_url="https://feinstaub.citysensor.de/api/getdata")

    sensor_data = sensor_api.fetch_data(
        params={"sensorid": "2446", "avg": "1", "span": "720"}
    )

    print(sensor_data["values"])

    # dict_keys(['sid', 'avg', 'span', 'start', 'count', 'values'])


if __name__ == "__main__":
    main()

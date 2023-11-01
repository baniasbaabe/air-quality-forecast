"""Running the feature pipeline.""."""
from dotenv import load_dotenv
from extract import SensorAPIClient
from load import Loader
from transform import Transformation


def main():
    """Main function for running the feature pipeline."""
    load_dotenv()

    # Extract
    sensor_api = SensorAPIClient(api_url="https://feinstaub.citysensor.de/api/getdata")
    sensor_data = sensor_api.fetch_data(
        params={"sensorid": "2446", "avg": "1", "span": "720"}
    )

    # Transform
    transformer = Transformation()
    df = transformer.run(data=sensor_data)

    # Load
    loader = Loader()
    loader.load_features(df=df)


if __name__ == "__main__":
    main()

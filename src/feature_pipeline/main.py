from extract import SensorAPIClient
from transform import Transformation
from load import Loader
from dotenv import load_dotenv


def main():
    load_dotenv()

    # Extract
    sensor_api = SensorAPIClient(api_url="https://feinstaub.citysensor.de/api/getdata")
    sensor_data = sensor_api.fetch_data(
        params={"sensorid": "2446", "avg": "1", "span": "720"}
    )

    # Transform
    transformer = Transformation()
    df = transformer.load_data_from_dict(data=sensor_data)
    df = transformer.preprocess_dataframe(df=df)
    df = transformer.resample_dataframe(df=df)

    # Load
    loader = Loader()
    loader.load_features(df=df)


if __name__ == "__main__":
    main()

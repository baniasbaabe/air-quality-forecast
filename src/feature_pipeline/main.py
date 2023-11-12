"""Running the feature pipeline.""."""
from pathlib import Path

import yaml
from dotenv import load_dotenv
from extract import SensorAPIClient
from load import Loader
from transform import Transformation


def main():
    """Main function for running the feature pipeline."""
    load_dotenv()
    CONFIG = yaml.safe_load(open(Path("config/config.yaml")))

    # Extract
    sensor_api = SensorAPIClient(api_url="https://feinstaub.citysensor.de/api/getdata")
    sensor_data = sensor_api.fetch_data(
        params={"sensorid": "stuttgart", **CONFIG["sensorapi"]}
    )

    # Transform
    transformer = Transformation()
    required_size = (
        CONFIG["conformal_prediction"]["n_windows"] + CONFIG["hyper_params"]["h"] + 1
    )
    df = transformer.run(data=sensor_data, required_size=required_size)

    # Load
    loader = Loader()
    loader.load_features(df=df)


if __name__ == "__main__":
    main()

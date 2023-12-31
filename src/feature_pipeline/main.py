"""Running the feature pipeline.""."""
from pathlib import Path

import yaml
from dotenv import load_dotenv
from extract import SensorAPIClient
from load import Loader
from transform import Transformation

from src import utils


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
    required_size = utils.calculate_required_size_for_conformal_prediction(
        h=CONFIG["hyper_params"]["h"],
        n_windows=CONFIG["conformal_prediction"]["n_windows"],
        cutoff_hours=CONFIG["train_test_split"]["cutoff_hours"],
    )
    df = transformer.run(data=sensor_data["sensordata"], required_size=required_size)

    # Load
    loader = Loader()
    loader.load_features(df=df)


if __name__ == "__main__":
    main()

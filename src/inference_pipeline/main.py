"""Running the inference pipeline."""

import joblib
import yaml
from dotenv import load_dotenv
from model_loader import ModelLoader
from prediction_database import MongoDBPredictionDatabase


def main():
    """Main function for running the inference pipeline."""
    load_dotenv()
    CONFIG = yaml.safe_load(open(r"config\config.yaml"))

    model_loader = ModelLoader(model_name=CONFIG["model"])
    model = model_loader.load_production_model()

    with open("./model.pkl", "rb") as f:
        model = joblib.load(f)

    predictions = model.predict(
        h=CONFIG["hyper_params"]["h"], level=CONFIG["conformal_prediction"]["levels"]
    )

    prediction_database = MongoDBPredictionDatabase()
    prediction_database.save_predictions(predictions)


if __name__ == "__main__":
    main()

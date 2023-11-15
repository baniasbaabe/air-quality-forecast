"""Running the inference pipeline."""

from pathlib import Path

import joblib
import yaml
from dotenv import load_dotenv
from model_loader import ModelLoader

from src.database import MongoDBDatabase


def main():
    """Main function for running the inference pipeline."""
    load_dotenv()
    CONFIG = yaml.safe_load(open(Path("config/config.yaml")))

    model_loader = ModelLoader(model_name=CONFIG["model"])
    model = model_loader.load_production_model()

    with open("./model.pkl", "rb") as f:
        model = joblib.load(f)

    predictions = model.predict(
        h=CONFIG["hyper_params"]["h"], level=CONFIG["conformal_prediction"]["levels"]
    )

    prediction_database = MongoDBDatabase()
    float_columns = predictions.select_dtypes(include="float").columns
    predictions[float_columns] = predictions[float_columns].applymap(
        lambda x: max(x, 0)
    )
    predictions.columns = [
        column.replace(CONFIG["model"], "Model") for column in predictions.columns
    ]
    print(predictions["ds"].max())
    prediction_database.save_predictions(predictions)


if __name__ == "__main__":
    main()

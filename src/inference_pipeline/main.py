"""
Running the inference pipeline.
"""

import joblib
from dotenv import load_dotenv
from model_loader import ModelLoader
from prediction_database import MongoDBPredictionDatabase


def main():
    """
    Main function for running the inference pipeline.
    """
    load_dotenv()

    model_loader = ModelLoader()
    model = model_loader.load_production_model()

    with open("./model.pkl", "rb") as f:
        model = joblib.load(f)

    predictions = model.predict(h=7, level=[80, 90])

    prediction_database = MongoDBPredictionDatabase()
    prediction_database.save_predictions(predictions)


if __name__ == "__main__":
    main()

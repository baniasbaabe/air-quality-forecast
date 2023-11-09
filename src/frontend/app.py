from dotenv import load_dotenv

import os
from src import utils
from pymongo.mongo_client import MongoClient
from src.inference_pipeline.prediction_database import MongoDBPredictionDatabase

load_dotenv()

project = utils.hopsworks_login()
fs = utils.hopsworks_get_feature_store(project)
fg = utils.hopsworks_get_feature_group(fs)
data = fg.select(["sid", "dt", "p1"]).read(read_options={"use_hive": True})

data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})


db = MongoDBPredictionDatabase()

db = MongoClient(
    f"mongodb+srv://{os.environ.get('MONGO_DB_USER')}:{os.environ.get('MONGO_DB_PASSWORD')}@airquality.boyq2bn.mongodb.net/?retryWrites=true&w=majority"
)
all_preds = db["AirQuality"]["AirQualityForecasts"].find({})
# print(list(all_preds))

import pandas as pd

all_preds = pd.DataFrame(list(all_preds))

all_preds.to_csv("all_preds.csv", index=False)

data.to_csv("data.csv", index=False)

# print(pd.merge(data, pd.DataFrame(list(all_preds)), on=["unique_id"]))

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.plot_date(
#     all_preds.query("unique_id == 140")["ds"],
#     all_preds.query("unique_id == 140")["AutoTheta"],
# )
# import plotly.express as px

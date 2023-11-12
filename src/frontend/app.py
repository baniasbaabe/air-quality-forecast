import os

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


from src import utils
from src.inference_pipeline.prediction_database import MongoDBPredictionDatabase


load_dotenv()


@st.cache_data()
def hopsworks_mongo_loading():
    project = utils.hopsworks_login()
    fs = utils.hopsworks_get_feature_store(project)
    fg = utils.hopsworks_get_feature_group(fs)
    data = fg.select(["sid", "dt", "p1"]).read(read_options={"use_hive": True})

    data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})

    db = MongoClient(
        f"mongodb+srv://{os.environ.get('MONGO_DB_USER')}:{os.environ.get('MONGO_DB_PASSWORD')}@airquality.boyq2bn.mongodb.net/?retryWrites=true&w=majority"
    )
    all_preds = db["AirQuality"]["AirQualityForecasts"].find({})

    all_preds = pd.DataFrame(list(all_preds))

    return data, all_preds


data, all_preds = hopsworks_mongo_loading()

selected_id = st.sidebar.selectbox("Select an Sensor ID", data["unique_id"].unique())

selected_data_forecast = all_preds[all_preds["unique_id"] == selected_id].sort_values(
    "ds"
)

# Filter the historic DataFrame based on the selected unique_id
selected_data_historic = data[data["unique_id"] == selected_id].sort_values("ds")

# Create the plot
fig = go.Figure()

# Plot the mean forecast line and prediction interval bands
fig.add_trace(
    go.Scatter(
        x=selected_data_forecast["ds"],
        y=selected_data_forecast["Model"],
        mode="lines",
        name="Model (Mean Forecast)",
    )
)

fig.add_trace(
    go.Scatter(
        x=selected_data_forecast["ds"],
        y=selected_data_forecast["Model-lo-90"],
        fill=None,
        mode="lines",
        line=dict(color="rgba(0, 0, 255, 0.3)"),
        name="Model (90% Prediction Interval)",
    )
)

fig.add_trace(
    go.Scatter(
        x=selected_data_forecast["ds"],
        y=selected_data_forecast["Model-hi-90"],
        fill="tonexty",
        mode="lines",
        line=dict(color="rgba(0, 0, 255, 0.3)"),
        name="Model (90% Prediction Interval)",
    )
)

# Plot the historic values
fig.add_trace(
    go.Scatter(
        x=selected_data_historic["ds"],
        y=selected_data_historic["y"],
        mode="markers",
        marker=dict(color="green", size=10),
        name="Historic Values (p1)",
    )
)

# Update layout
fig.update_layout(
    title=f"Forecasts and Historic Values for unique_id: {selected_id}",
    xaxis_title="Date",
    yaxis_title="Values",
    showlegend=True,
)

st.plotly_chart(fig, use_container_width=False)

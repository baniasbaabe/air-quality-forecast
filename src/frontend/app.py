#TODO: Theming, Description
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src import utils
from src.database import MongoDBDatabase

st.set_page_config(page_title="Air Quality Forecasting", page_icon="https://cdn-icons-png.flaticon.com/512/3260/3260792.png", layout="wide")

load_dotenv()

st.title("Air Quality Forecasting")

st.markdown("This is an End-to-End ML Project for the Air Quality Forecasting Project. \
    The project fetches data from (Feinstaub-Citysensor)[https://feinstaub.citysensor.de]. The model predicts the amount of PM10 in the air for the next 24 hours \
    for every Sensor ID in Stuttgart, Germany. There are over 100 Sensor IDs located in Stuttgart. See here for more information: (GitHub)[https://github.com/baniasbaabe/air-quality-forecast]")


@st.cache_data(ttl="30min")
def hopsworks_mongo_loading():
    project = utils.hopsworks_login()
    fs = utils.hopsworks_get_feature_store(project)
    fg = utils.hopsworks_get_feature_group(fs)
    data = fg.select(["sid", "dt", "p1"]).read(read_options={"use_hive": True})

    data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})

    db = MongoDBDatabase().connect()

    all_preds = db["AirQuality"]["AirQualityForecasts"].find({})

    all_preds = pd.DataFrame(list(all_preds))

    return data, all_preds


data, all_preds = hopsworks_mongo_loading()

selected_id = st.sidebar.selectbox("Select an Sensor ID. Sensor IDs are described (here)[https://feinstaub.citysensor.de]", data["unique_id"].unique())

selected_data_forecast = all_preds[all_preds["unique_id"] == selected_id].sort_values(
    "ds"
)

selected_data_historic = data[data["unique_id"] == selected_id].sort_values("ds")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=selected_data_forecast["ds"],
        y=selected_data_forecast["Model"],
        mode="lines",
        name="Mean Prediction",
        
    )
)

fig.add_trace(
    go.Scatter(
        x=selected_data_forecast["ds"],
        y=selected_data_forecast["Model-lo-90"],
        fill=None,
        mode="lines",
        line=dict(color="rgba(0, 0, 255, 0.3)"),
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter(
        x=selected_data_forecast["ds"],
        y=selected_data_forecast["Model-hi-90"],
        fill="tonexty",
        mode="lines",
        line=dict(color="rgba(0, 0, 255, 0.3)"),
        name="90% Prediction Interval",
    )
)

fig.add_trace(
    go.Scatter(
        x=selected_data_historic["ds"],
        y=selected_data_historic["y"],
        marker=dict(color="black", size=10),
        line_shape='linear',
        name="Historic Values for P10",
    )
)

fig.update_layout(
    title=f"Forecasts and Historic Values for Sensor ID: {selected_id}",
    xaxis_title="Date",
    yaxis_title="Values",
    showlegend=True,
)

st.plotly_chart(fig, use_container_width=False)

from dotenv import load_dotenv

import os
from src import utils
from pymongo.mongo_client import MongoClient
from src.inference_pipeline.prediction_database import MongoDBPredictionDatabase

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

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

import pandas as pd

all_preds = pd.DataFrame(list(all_preds))

import plotly.graph_objects as go


# fig = go.Figure()

# # Plot the mean forecast line
# fig.add_trace(
#     go.Scatter(
#         x=all_preds["ds"],
#         y=all_preds["Model"],
#         mode="lines",
#         name="Model (Mean Forecast)",
#     )
# )

# fig.add_trace(
#     go.Scatter(
#         x=all_preds["ds"],
#         y=all_preds["Model-lo-90"],
#         fill=None,
#         mode="lines",
#         line=dict(color="rgba(0, 0, 255, 0.3)"),
#         showlegend=False,
#     )
# )

# fig.add_trace(
#     go.Scatter(
#         x=all_preds["ds"],
#         y=all_preds["Model-hi-90"],
#         fill="tonexty",
#         mode="lines",
#         line=dict(color="rgba(0, 0, 255, 0.3)"),
#         name="90% Prediction Interval",
#     )
# )
# fig.update_layout(
#     title=f"Forecasts for unique_id: 140",
#     xaxis_title="Date",
#     yaxis_title="P10",
#     showlegend=True,
# )

# fig.add_trace(go.Scatter(x=data["ds"], y=data["y"], mode="lines", name="Historic"))

app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    [
        html.Label("Select unique_id:"),
        dcc.Dropdown(
            id="unique-id-dropdown",
            options=[
                {"label": uid, "value": uid} for uid in all_preds["unique_id"].unique()
            ],
            value=all_preds["unique_id"].iloc[0],
        ),
        dcc.Graph(id="forecast-plot"),
    ]
)


# Define callback to update the figure based on the selected unique_id
@app.callback(Output("forecast-plot", "figure"), [Input("unique-id-dropdown", "value")])
def update_figure(selected_unique_id):
    # Filter the forecast DataFrame based on the selected unique_id
    selected_data_forecast = all_preds[
        all_preds["unique_id"] == selected_unique_id
    ].sort_values("ds")

    # Filter the historic DataFrame based on the selected unique_id
    selected_data_historic = data[data["unique_id"] == selected_unique_id].sort_values(
        "ds"
    )

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
        title=f"Forecasts and Historic Values for unique_id: {selected_unique_id}",
        xaxis_title="Date",
        yaxis_title="Values",
        showlegend=True,
    )

    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

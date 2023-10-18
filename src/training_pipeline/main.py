import pandas as pd
from dotenv import load_dotenv
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta
from statsforecast.utils import ConformalIntervals


def main():
    load_dotenv()

    # project = hopsworks_login()
    # fs = hopsworks_get_feature_store(project)
    # fg = hopsworks_get_feature_group(fs)
    # data = fg.select_all().read()

    # data = data[data["dt"] <= datetime.datetime.now() - pd.to_timedelta("7day")]

    # print(data)

    data = pd.read_csv("data.csv")

    data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})

    model = AutoTheta(prediction_intervals=ConformalIntervals(h=7, n_windows=3))

    models = [model]

    sf = StatsForecast(
        df=data,
        models=models,
        freq="H",
    )

    levels = [80, 90]

    forecasts = sf.forecast(h=7, level=levels)
    forecasts = forecasts.reset_index()
    forecasts.head()

    print(forecasts.head())
    print(data["ds"].max())


if __name__ == "__main__":
    main()

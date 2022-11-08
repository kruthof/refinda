import pandas as pd
import numpy as np
import datetime


def get_ticker_columns(data):
    """
    converts refinda data into table with index ==date, c
    columuns = close price per ticker

    :param data: refinda dataset
    :return: dataframe close prices per date and ticker
    """
    output = None
    for ticker in np.unique(data["tic"]):
        if output is None:
            tick_data = data.loc[data["tic"] == ticker, ["close", "date"]].set_index(
                "date"
            )
            output = pd.DataFrame({"date": tick_data.index, ticker: tick_data["close"]})
            output.set_index("date", inplace=True)

        else:
            tick_data = data.loc[data["tic"] == ticker, ["close", "date"]].set_index(
                "date"
            )
            output[ticker] = tick_data["close"]
    return output


def get_ticker_columns_returns(data):
    """
    Function converts refinda data into table with index ==date, c
    columuns = close price per ticker

    :param data: refinda dataset
    :return: dataframe returns per date and ticker
    """
    return get_ticker_columns(data).pct_change()


def strategy_preprocessor(data, start, end, window):

    """
    Function returns data sliced by start and end date plus time of lenght window before
    start date
    @param data: column transformed refinda data
    @param start: string first date at which weights are needed
    @param end: string last date for which weights are needed
    @param window: int length of time used for subsequent strategy
    """

    # esure window is int
    window = int(window)
    # check if end date exist in data, previously available day otherwise
    start = get_startDate_days(data, end_date=start, delta=0)
    end = get_endDate_days(data, start_date=end, delta=0)

    data = data.reset_index()

    start_idx = data.loc[data["date"] == start, :].index.values[0]
    # check if end date exist in data, next available day otherwise
    end_idx = data.loc[data.date == end, :].index.values[0]

    data = data.iloc[start_idx - window : end_idx + 1]
    return data


def get_endDate_days(data, start_date, delta):
    end = pd.to_datetime(start_date) + datetime.timedelta(days=delta)
    # check if end is in data, else add one additional day
    while end not in pd.to_datetime(data.index):
        end += datetime.timedelta(days=1)
    return end.strftime("%Y-%m-%d")


def get_startDate_days(data, end_date, delta):
    start = pd.to_datetime(end_date) - datetime.timedelta(days=delta)
    # check if end is in data, else add one additional day
    while start not in pd.to_datetime(data.index):
        start -= datetime.timedelta(days=1)
    return start.strftime("%Y-%m-%d")

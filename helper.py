from datetime import datetime
import pandas as pd


def str_to_float(data, date="Date"):
    """
    Converts string data float excep Date variable
    """
    for col in data.columns:
        if col != date:
            data[col] = [float(str(x).strip()) for x in data[col].values]
        else:
            data[col] = [
                datetime.strptime(str(x).strip(), "%Y%m%d") for x in data[col].values
            ]
    return data


def convert_prices(dailyReturns, start=100):
    """
    Function converts return values to prices using starting value of start

    @param dailyReturns vector of returns

    @return vector of prices
    """

    df = pd.DataFrame({"dailyReturns": dailyReturns})
    df["close"] = start * (1 + df["dailyReturns"] / 100).cumprod()

    return df["close"]

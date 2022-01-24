import ta
import pandas as pd
import numpy as np


def indicators_tech_closePriceOnly(price, date, name):
    """
    Function calculates 31 technical indicators for a price vector

    @param price vector of price
    @param date vector date corresponding to prices
    @param name str name of company/portfolio


    @return df dataframe with price, company/portfolio name and tech indicators
    """
    # convert vector to DataFrame
    df = pd.DataFrame({"price": price})
    df["date"] = date
    df["name"] = name.strip()

    # add columns with indicators
    # Momentum indicators

    PPO = ta.momentum.PercentagePriceOscillator(price)
    df["ppo"] = PPO.ppo()
    df["ppo_hist"] = PPO.ppo_hist()
    df["ppo_signal"] = PPO.ppo_signal()

    # Rate of Change (ROC)
    df["roc"] = ta.momentum.ROCIndicator(price).roc()
    # Relative Strength Index (RSI)
    df["rsi"] = ta.momentum.RSIIndicator(price).rsi()

    # Stochastic RSI
    StockRsi = ta.momentum.StochRSIIndicator(price)
    df["rsi_stoch"] = StockRsi.stochrsi()
    df["rsi_stochrsi_d"] = StockRsi.stochrsi_d()
    df["rsi_stochrsi_k"] = StockRsi.stochrsi_k()

    # True strength index (TSI)
    df["tsi"] = ta.momentum.TSIIndicator(price).tsi()

    # Kaufman’s Adaptive Moving Average (KAMA)
    df["kama"] = ta.momentum.kama(price)

    # Volatility Indicators

    indicator_bb = ta.volatility.BollingerBands(price)
    df["bollinger_mavg"] = indicator_bb.bollinger_mavg()
    df["bollinger_hband"] = indicator_bb.bollinger_hband()
    df["bollinger_lband"] = indicator_bb.bollinger_lband()

    df["bollinger_pband"] = indicator_bb.bollinger_pband()
    df["bollinger_wband"] = indicator_bb.bollinger_wband()

    # Ulcer Index
    df["ulcer"] = ta.volatility.UlcerIndex(price).ulcer_index()

    # Volatility

    # Aroon Indicator
    indocator_aroon = ta.trend.AroonIndicator(price)
    df["aroon_down"] = indocator_aroon.aroon_down()
    df["aroon_ind"] = indocator_aroon.aroon_indicator()
    df["aroon_up"] = indocator_aroon.aroon_up()

    # Detrended Price Oscillator (DPO)
    df["dpo"] = ta.trend.DPOIndicator(price).dpo()

    # EMA - Exponential Moving Average
    df["ema"] = ta.trend.EMAIndicator(price).ema_indicator()

    # KST Oscillator (KST Signal)
    indicator_kst = ta.trend.KSTIndicator(price)
    df["kst"] = indicator_kst.kst()
    df["kst_diff"] = indicator_kst.kst_diff()
    df["kst_sig"] = indicator_kst.kst_sig()

    # MACD
    indicator_macd = ta.trend.MACD(price)
    df["macd"] = indicator_macd.macd()
    df["macd_diff"] = indicator_macd.macd_diff()
    df["macd_signal"] = indicator_macd.macd_signal()

    # SMA Simple MA
    df["sma"] = ta.trend.SMAIndicator(price, window=20).sma_indicator()

    # Schaff Trend Cycle
    df["stc"] = ta.trend.STCIndicator(price).stc()

    # Trix
    df["trix"] = ta.trend.TRIXIndicator(price).trix()

    # WMA Weighted MA
    df["wma"] = ta.trend.WMAIndicator(price).wma()

    # return dataframe
    return df


def french49_generate_indicators_closedPrices(data, companies, date="Date"):
    """
    Function applied in indicators_tech_closePriceOnly function to French dataset. Prices
    in the original dataset are stored using portfolio name, e.g. 'Smoke' = 1,0.5,...

    @param data dataframe with price vector with name E companies.
    @param companies list names of portfolios
    @param date str name of date variable in data

    return df dataframe featuring all
    """

    # init empty dataframe
    df = None

    for symbol in companies:
        if df is None:
            df = indicators_tech_closePriceOnly(data[symbol], data[date], name=symbol)
        else:
            df = df.append(
                indicators_tech_closePriceOnly(data[symbol], data[date], name=symbol)
            )

    # subset data no older then 1970, as only then all indicators feature non-NAN values
    df = (
        df[(df["date"] > "1970-01-01") & (df["date"] <= "2021-11-30")]
        .sort_values(by="date")
        .reset_index()
    )
    df = df.iloc[:, 1:]
    print(int(np.mean(df.price)))

    # check if sum of prices equal number identified during initial setup
    if int(np.mean(df.price)) != 263697:
        raise NameError("Error 001: dataset versions do not match")
        # return None

    # no more NA should be present at this stage

    if df.iloc[1:].isnull().values.any():
        raise NameError("Error 002: Unexpected NA values!")

    return df

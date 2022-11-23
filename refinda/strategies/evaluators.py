import pandas as pd
from pyfolio import timeseries
import numpy as np
from refinda.helper.helper_general import _rolling_apply

def get_summary(returns_vector, nan="remove"):
    """
    Providing summary statistics for a given return vector

    @param returns_vector pandas series

    @return pandas series summary statistics
    """

    if nan is "remove":
        returns_vector = returns_vector[returns_vector.notna()]
    elif nan is "zero":
        returns_vector = returns_vector.fillna(0)
    else:
        raise KeyError("missing value strategy not available")

    perf_func = timeseries.perf_stats
    return perf_func(returns_vector)

def compare_summary(return_dict=[]):
    '''
    applies get_summary to a dict of return series and
    shows results in one dataframe
    @param return_dict dict of return series

    @return dataframe of summary statistics
    '''
    overview_df = None
    for return_series in return_dict.keys():
        _returns = get_summary(return_dict[return_series])
        if overview_df is None:
            overview_df = pd.DataFrame({'KPI':_returns.index,
                                        return_series:_returns.values})
        else:
            overview_df[return_series] = _returns.values
    return overview_df.set_index('KPI')

def sharp_ratio(data,return_freq = 'daily'):
    '''
    Function returns sharop value of a given time series

    @param data series of (risk adjusted) returns
    @return float sharpvalue
    '''
    _adjustments = {"daily": 252,
                  "weekly": 52,
                  "monthly": 12}
    try:
        adjustment = _adjustments[return_freq]
    except:
        raise KeyError('not a valid return frequency')
    return np.multiply(
            np.divide(np.mean(data),np.std(data))
            , np.sqrt(adjustment)
            )

def sharp_ratio_rolling(data,window,return_freq):
    '''
    Function calculates sharp ratio using rolling window
    @param data array return data
    @param window int rolling window
    @param return_freq list sharp ration adjustments

    @return array
    '''
    #store data
    date=data.index
    #get sharp ratios
    sharp_value = _rolling_apply(data, sharp_ratio, window=window, args=return_freq)
    #return dataframe with date as index
    return pd.DataFrame({'date':date,'sharp_ratios':sharp_value}).set_index('date').iloc[window:]

import pandas as pd
from pyfolio import timeseries
import numpy as np
from refinda.helper.helper_general import _rolling_apply
import scipy
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
    Function returns annualized sharp value of a given time series

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

def kurtosis(data,return_freq='daily'):
    '''
    Function calculates kurtosis for a given series and annualizes
    the result.
    https://quant.stackexchange.com/questions/3956/how-to-annualize-skewness-and-kurtosis-based-on-daily-returns

    @param data series of returns
    @param return_freq str indicating period for returns

    @return annualized kurtosis
    '''

    _adjustments = {"daily": 252,
                  "weekly": 52,
                  "monthly": 12}

    kurtosis = scipy.stats.kurtosis(data, nan_policy='omit')
    return kurtosis / _adjustments[return_freq]

def skewness(data,return_freq='daily'):
    '''
    Function calculates skewness for a given series and annualizes
    the result.
    https://quant.stackexchange.com/questions/3956/how-to-annualize-skewness-and-kurtosis-based-on-daily-returns

    @param data series of returns
    @param return_freq str indicating period for returns

    @return annualized skeqness
    '''

    _adjustments = {"daily": 252,
                  "weekly": 52,
                  "monthly": 12}

    skewness = scipy.stats.skew(data[-np.isnan(data)],nan_policy='omit')
    return skewness / np.sqrt(_adjustments[return_freq])



def kurtosis_ratio_rolling(data,window,return_freq):
    '''
    Function calculates kurtosis using rolling window
    @param data array return data
    @param window int rolling window
    @param return_freq list skewness adjustments

    @return array rolling kurtosis
    '''
    #store data
    date=data.index
    #get sharp ratios
    _kurtosis = _rolling_apply(data, kurtosis, window=window, args=return_freq)
    #return dataframe with date as index
    return pd.DataFrame({'date':date,'kurtosis':_kurtosis}).set_index('date').iloc[window:]

def calculate_annualized_mean(data,return_frequency='daily'):
    '''
    calculate adjusted mean
    @param data dataframe return data
    @param return_frequency str frequency of returns

    @return annualized mean
    '''
    _adjustments = {"daily": 252,
                    "weekly": 52,
                    "monthly": 12}
    return np.mean(data) * _adjustments[return_frequency]

def mean_returns_rolling(data,window,return_freq=['daily']):
    '''
    Function calculates mean returns using rolling window
    @param data array return data
    @param window int rolling window
    @param return_freq list skewness adjustments

    @return array rolling kurtosis
    '''
    _adjustments = {"daily": 252,
                  "weekly": 52,
                  "monthly": 12}
    #store data
    date=data.index
    #get sharp ratios
    _returns = _rolling_apply(data, calculate_annualized_mean, window=window, args=return_freq) #* _adjustments[return_freq]
    #return dataframe with date as index
    return pd.DataFrame({'date':date,'returns':_returns}).set_index('date').iloc[window:]

def skewness_ratio_rolling(data,window,return_freq):
    '''
    Function calculates skewness using rolling window
    @param data array return data
    @param window int rolling window
    @param return_freq list skewness adjustments

    @return array rolling skewness
    '''
    #store data
    date=data.index
    #get sharp ratios
    _skewness = _rolling_apply(data, skewness, window=window, args=return_freq)
    #return dataframe with date as index
    return pd.DataFrame({'date':date,'skewness':_skewness}).set_index('date').iloc[window:]

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

def expected_shortfall(data,cl=0.01):
    '''
    Function calculates the  expected shortfall for a given percentile
    @param data array datapoints
    @param cl float confidence level for calculating ES
    '''
    data=data[-np.isnan(data)]#ignore nan
    sorted_list = sorted(data.values)
    return np.mean(sorted_list[0:int(len(sorted_list) * cl)])
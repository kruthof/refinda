import pandas as pd
import numpy as np

def get_covariance(data, window):
    """
    Function generates covariance matrix based on data of shape(window,...)
    Covariance is "out-of-sample", it includes not the latest date
    e.g. the covariance for 01.01.2022 is based on data until 31.12.2021.

    @param: data dataframe column based data
    @param: window int integer indicating the length of the window, for which the
            cov matrix should be based on
    @return cov_list dataframe with array of covariances per date. date as index
    """
    cov_list = []

    date = data["date"].copy()

    for i in range(data.shape[0] - window):
        cov_list.append(data.iloc[i : i + window].cov().values)

    cov_list = pd.DataFrame(
        {
            "cov": cov_list[1 : len(cov_list) + 1],
            "date": date[1 + window : len(date) + 1],
        }
    )

    return cov_list.set_index("date")

def _rolling_apply(df, fun, window,args=[]):
    '''
    Function for applying functions to
    timeseries with rolling window
    @param df dataframe refinda column bases
    @param fun function to be applied
    @param window int rolling window

    @return object
    '''
    prepend = [None] * (window)
    end = len(df) - window
    mid = map(lambda start: fun(df.iloc[start:start + window],*args), np.arange(0,end))
    #last =  fun(df.iloc[end:])
    #first =  fun(df.iloc[0:window])

    return [*prepend, *mid]

def get_rf():
    '''
    Function loads rf rates, transforms date column and set it as index

    @return df dataframe rf rates
    '''
    df = pd.read_csv('./datasets/rf_rates.csv')

    df['date'] = pd.to_datetime(df.date, format="%Y%m%d")
    df['date'] = [x.strftime("%Y-%m-%d") for x in df['date']]
    df.set_index('date',inplace=True)
    return df/100 #convert to percentage

def rf_adjustment(data):
    '''
    Function provides risk free adjusted returns
    @param data dataframe with index column date and returns with column name returns
    @returns dataframe df risk adjusted returns with index = date
    '''
    rf = get_rf()
    df = pd.merge(data.returns, rf, right_index=True, left_index=True)
    return df.iloc[:,0] - df.iloc[:,1]



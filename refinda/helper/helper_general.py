import pandas as pd

def get_covariance(data, window):
    '''
    Function generates covariance matrix based on data of shape(window,...)
    Covariance is "out-of-sample", it includes not the latest date
    e.g. the covariance for 01.01.2022 is based on data until 31.12.2021.

    @param: data dataframe column based data
    @param: window int integer indicating the length of the window, for which the
            cov matrix should be based on
    @return cov_list dataframe with array of covariances per date. date as index
    '''
    cov_list = []

    date = data['date'].copy()

    for i in range(data.shape[0] - window):
        cov_list.append(data.iloc[i:i + window].cov().values)

    cov_list = pd.DataFrame({'cov': cov_list[1:len(cov_list) + 1], 'date': date[1 + window:len(date) + 1]})

    return cov_list.set_index('date')


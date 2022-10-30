import pandas as pd
import numpy as np
def get_ticker_columns(data):
    '''
    converts refinda data into table with index ==date, c
    columuns = close price per ticker

    :param data: refinda dataset
    :return: dataframe close prices per date and ticker
    '''
    output = None
    for ticker in np.unique(data['tic']):
        if output is None:
            tick_data = data.loc[data['tic']==ticker,['close','date']].set_index('date')
            output = pd.DataFrame({'date': tick_data.index,ticker:tick_data['close']})
            output.set_index('date',inplace=True)

        else:
            tick_data = data.loc[data['tic']==ticker,['close','date']].set_index('date')
            output[ticker] = tick_data['close']
    return output
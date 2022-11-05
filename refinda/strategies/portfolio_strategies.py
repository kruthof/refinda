import pandas as pd
import numpy as np
import scipy.optimize as sco

def strategy_1n(data,funds = 1000000, transaction_costs=0.001):
    '''
    Generates 1/n strategy with re-allocating funds equally across portfolios
    every timestamp.

    :param data: refinda dataframe each column = close prices for each ticker, index = timestamp.
    :param funds: int, starting funds, default =100
    :return: Dataframe funds per timestamp and absolute amount of shares per portfolio and timestamp
    '''
    #get number of portfolios
    n_portfolios=data.shape[1]
    #calculate funds per portfolio
    funds_asset = funds / n_portfolios
    #initiate dataframe
    funds_investment= pd.DataFrame({'date':data.index,'funds':None,'assets':None,'turnover':None,'transaction_costs':None,'returns':None})

    #provide strategy information
    print(f'1/N strategy with initial funding: {funds} and transaction costs: {transaction_costs*100}%.')
    #loop through dataframe
    for i in range(data.shape[0]):
        #calculate stocks per portfolio

        #calculate turnover and transaction costs
        if i>0:
            funds = (data.iloc[i]* assets.values).sum()
            #update funds per portfolio
            funds_asset = funds/n_portfolios
            #calculate assets delta from t-1 to t
            assets_new = funds_asset / data.iloc[i]

            assets_delta = np.abs(assets_new -  assets)
            #calculate turnover
            turnover = np.sum(assets_delta * data.iloc[i])

        else:
            assets =  funds_asset / data.iloc[i]
            turnover = np.sum(funds)
            assets_new = assets
            funds = (data.iloc[i]* assets.values).sum()
            #update funds per portfolio
            funds_asset = funds/n_portfolios


        funds_investment.loc[i,'turnover'] = turnover
        funds_investment.loc[i,'transaction_costs'] = turnover * transaction_costs
        #calculate fundings for next timestep
        #print(f'assets are {assets_new.values}')
        #wrtie to dataframe
        funds_investment.loc[i,'funds'] = funds
        #absolut measures of stocks per portolio
        assets = assets_new.copy()
        #assets_new = funds_asset / data.iloc[i]

        funds_investment.loc[i,'assets'] = assets_new.values

    funds_investment['returns']  = funds_investment.funds.pct_change()
    return funds_investment.set_index('date')

def get_minVar_Portfolio(data,window,rf_exclude=True):
    '''
    Function generates covariance matrix based on data of shape(window,...)
    Covariance is "out-of-sample", it includes not the latest date
    e.g. the covariance for 01.01.2022 is based on data until 31.12.2021.

    @param: data dataframe column based data
    @param: window int integer indicating the length of the window, for which the
            cov matrix should be based on
    @return cov_list dataframe with array of covariances per date. date as index
    '''

    date = np.array(data['date'].copy())

    if rf_exclude:
        data = data.iloc[:,(data.columns != 'RF') & (data.columns != 'date')]
    else:
        data = data.iloc[:,data.columns != 'date']

    #calculate percentage change
    data = data.pct_change()
    #create empty df
    minVar_weights = pd.DataFrame()

    #Prepare optimizer
    #  create a sequence of (min, max) pairs
    bounds = tuple((0, 1) for w in range(len(data.columns)))
    # anonymous lambda function
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Repeat the list with the value (1 / 12) 12 times, and convert list to array
    equal_weights = np.array([1 / len(data.columns)] * len(data.columns))
    #loop through each observation and calculate min var weights using the last n=window observations
    for i in range(data.shape[0]-window):
        # Minimization results
        _minVar = sco.minimize(
          # Objective function
          fun = lambda weights: np.sqrt(np.transpose(weights) @ (data.iloc[i:i+window].cov() * 253/window) @ weights),
          # Initial guess, which is the equal weight array
          x0 = equal_weights,
          method = 'SLSQP',
          bounds = bounds,
          constraints = constraints
        )['x']
        #append result to dataframe
        minVar_weights = minVar_weights.append(pd.Series(_minVar),ignore_index=True)
        #returns.append(np.dot(_returns,_minVar).sum())


    minVar = minVar_weights.iloc[0:len(date)]
    minVar.columns = data.columns #rename columns
    minVar['date']= np.array(date[0+window:len(date) ]) #include date again


    return minVar.set_index('date') #set date as index and return
import pandas as pd
import numpy as np

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

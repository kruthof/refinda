import pandas as pd
import numpy as np
import scipy.optimize as sco
from refinda.helper.transformation import get_ticker_columns
from refinda.helper.helper_general import _rolling_apply


class portfolioStrategies:
    def __init__(self, data, window):
        self.data = data
        self.window = window
        self.weights_minVar = None
        self.weights_maxSharp = None
        self.rf_exclude = True
        self.transaction_costs = 0.001
        self.funds = 1000000
        try:
            self.data = get_ticker_columns(self.data).set_index("date")
        except:
            pass

    def strategy_1n(self):
        """
        Generates 1/n strategy with re-allocating funds equally across portfolios
        every timestamp.

        :param data: refinda dataframe each column = close prices for each ticker, index = timestamp.
        :param funds: int, starting funds, default =100
        :return: Dataframe funds per timestamp and absolute amount of shares per portfolio and timestamp
        """
        # transform reinfda dataset to column data

        data = self.data.iloc[self.window :].set_index(
            "date"
        )  # no need for look-back window for 1/n
        # get number of portfolios
        n_portfolios = data.shape[1]
        # calculate funds per portfolio
        funds_asset = self.funds / n_portfolios
        # initiate dataframe
        funds_investment = pd.DataFrame(
            {
                "date": data.index,
                "funds": None,
                "assets": None,
                "turnover": None,
                "transaction_costs": None,
                "returns": None,
            }
        )

        # provide strategy information
        print(
            f"1/N strategy with initial funding: {self.funds } and transaction costs: {self.transaction_costs *100}%."
        )
        # loop through dataframe
        for i in range(data.shape[0]):
            # calculate stocks per portfolio

            # calculate turnover and transaction costs
            if i > 0:
                funds = (data.iloc[i] * assets.values).sum()
                # update funds per portfolio
                funds_asset = funds / n_portfolios
                # calculate assets delta from t-1 to t
                assets_new = funds_asset / data.iloc[i]

                assets_delta = np.abs(assets_new - assets)
                # calculate turnover
                turnover = np.sum(assets_delta * data.iloc[i])
                funds_investment.loc[i, "assets_delta"] = [assets_delta]
            else:
                assets = funds_asset / data.iloc[i]
                turnover = np.sum(self.funds)
                assets_new = assets
                funds = (data.iloc[i] * assets.values).sum()
                # update funds per portfolio
                funds_asset = funds / n_portfolios
                funds_investment.loc[i, "assets_delta"] = [0]
            funds_investment.loc[i, "turnover"] = turnover
            funds_investment.loc[i, "transaction_costs"] = (
                turnover * self.transaction_costs
            )
            
            # calculate fundings for next timestep
            # print(f'assets are {assets_new.values}')
            # wrtie to dataframe
            funds_investment.loc[i, "funds"] = funds
            # absolut measures of stocks per portolio
            assets = assets_new.copy()
            # assets_new = funds_asset / data.iloc[i]

            funds_investment.loc[i, "assets"] = assets_new.values

        funds_investment["returns"] = funds_investment.funds.pct_change()
        return funds_investment.set_index("date")

    def strategy_minimum_variance(self):
        """
        Function takes takes min_var weights and applies it to dataset

        @return df dataframe featuring returns, turnover, and TC or min_var strategy
        """

        if self.weights_minVar is None:
            self.weights_minVar = self.rolling_apply(self._minVariance_optimizer)
        return self.apply_strategy(self.weights_minVar)

    def strategy_maximum_sharp(self):
        """
        Function takes takes min_var weights and applies it to dataset

        @return df dataframe featuring returns, turnover, and TC or min_var strategy
        """

        if self.weights_maxSharp is None:
            self.weights_maxSharp = self.rolling_apply(self._maxSharpValue_optimizer)
        return self.apply_strategy(self.weights_maxSharp)

    def apply_strategy(self, weights):
        """
        Functions calculates returns, turnover and transactino fees
        for min variance portfolio strategy

        @param weights dataframe weight df from get_PortfolioWeights function
        @param data dataframe column based price df from strategy_preprocessor function
        @param window int window size used for calculating weights
        """

        data = self.data.iloc[self.window : self.data.shape[0]].set_index("date")
        cumreturns = (
            (data.pct_change() * weights).apply(np.sum, axis=1).add(1).cumprod().sub(1)
        )
        funds = self.funds * cumreturns + self.funds
        prices = data.iloc[self.window : data.shape[0]]
        transaction = weights - weights.shift(1)
        transaction.iloc[0] = weights.iloc[0]
        turnover = np.abs(transaction).multiply(funds, axis=0).apply(np.sum, axis=1)
        transaction_fees = turnover * self.transaction_costs
        output = pd.DataFrame(
            {
                "date": data.index,
                "funds": funds,
                "turnover": turnover,
                "transaction_costs": transaction_fees,
                "returns": funds.pct_change(),
            }
        )

        return output

    def rolling_apply(self, function):
        """
        Function that applies optimizer for each rolling window
        @param object function to be applied

        @return weights dataframe weights for each timestep and asset
        """

        if self.rf_exclude:
            data = self.data.iloc[
                :, (self.data.columns != "RF") & (self.data.columns != "date")
            ]
        else:
            data = self.data.iloc[:, self.data.columns != "date"]
        # calculate percentage change
        data = data.pct_change().fillna(0)
        # apply optimizer
        weights = _rolling_apply(data, lambda x: function(x), self.window)
        # convert to dataframe
        weights = pd.DataFrame(weights[self.window : len(weights)])
        # rename columns
        weights.columns = data.columns
        # add date variable
        weights["date"] = self.data.iloc[self.window : self.window + weights.shape[0]][
            "date"
        ].values
        return weights.set_index("date")  # return date indexed weights

    def df_postProcessor(self, weights, data, date):
        """
        helper function for get_PortfolioWeights()
        slicing, renaming columns and indexing of weighting vecor
        """
        output = weights.iloc[0 : len(date)]
        output.columns = data.columns  # rename columns
        output["date"] = np.array(
            date[0 + self.window : len(date)]
        )  # include date again
        return output.set_index("date")  # set date as index

    def _minVariance_optimizer(self, data):
        """
        Optimizer for Min Variance Portfolio
        """
        #  create a sequence of (min, max) pairs
        data = pd.DataFrame(data)
        bounds = tuple((0, 1) for w in range(len(data.columns)))
        # anonymous lambda function
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # Repeat the list with the value (1 / 12) 12 times, and convert list to array
        equal_weights = np.array([1 / len(data.columns)] * len(data.columns))
        return sco.minimize(
            # Objective function
            fun=lambda weights: np.sqrt(
                np.transpose(weights) @ (data.cov() * 253) @ weights
            ),
            # Initial guess, which is the equal weight array
            x0=equal_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )["x"]

    def _maxSharpValue_optimizer(self, data):
        """
        Optimizer for max Sharp value portfolio
        """
        # Prepare optimizer
        #  create a sequence of (min, max) pairs
        bounds = tuple((0, 1) for w in range(len(data.columns)))
        # anonymous lambda function
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # Repeat the list with the value (1 / 12) 12 times, and convert list to array
        equal_weights = np.array([1 / len(data.columns)] * len(data.columns))
        return sco.minimize(
            # Objective function
            fun=lambda weights: -(
                self.portfolio_returns(weights, data, self.window)
                / self.portfolio_sd(weights, data, self.window)
            ),
            # Initial guess, which is the equal weight array
            x0=equal_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )["x"]

    def portfolio_returns(self, weights, data, calendarAdjust=253):
        """
        Helper function to calculate returns
        """
        return (np.sum(np.mean(data) * weights)) * calendarAdjust

    def portfolio_sd(self, weights, data, calendarAdjust=253):
        """
        Helper function zo calculate standard deviation
        """
        return np.sqrt(np.transpose(weights) @ (data.cov() * calendarAdjust) @ weights)

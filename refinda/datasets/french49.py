from refinda.preprocessors.indicators import *
import warnings
from refinda.preprocessors.helper import *
from tqdm import tqdm


def french49(file=None):
    warnings.filterwarnings("ignore")

    with tqdm(total=100) as pbar:

        if file == None:
            data = pd.read_csv(
                "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip",
                compression="zip",
                skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            )
        else:
            data = pd.read_csv(
                file,
                skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            )

        # update progress bar
        pbar.update(50)

        # exclude last entry indicating coopyrights
        data = data.iloc[0 : data.shape[0] - 1]

        # create date variable
        data["Date"] = data.iloc[:, 0]
        # data['Index'] =  [i for i in range(data.shape[0])]

        # drop unnamed column
        data = data.drop(["Unnamed: 0"], axis=1)

        # select equally weighted
        data = data.iloc[25128 : data.shape[0]]
        data = data.reset_index()

        data = str_to_float(data, date="Date")

        # update progress bar
        pbar.update(10)

        # Restric data to >= 1969-07-01 as this is the first months with data availabiltiy for all 49 portfolios
        data = data.loc[
            (data["Date"] >= "1969-07-01") & (data["Date"] <= "2021-11-30"), :
        ]

        # apply convert_prices function to convert return values to prices using 100 as start
        df = data.iloc[:, 1 : data.shape[1] - 1].apply(convert_prices, axis=0)
        # add date variable again
        df["Date"] = data["Date"]

        # get portfolio names
        portfolios = df.iloc[:, 0 : df.shape[1] - 1].columns

        # calculate indicstors
        df = french49_generate_indicators_closedPrices(df, portfolios)

        # update progress bar
        pbar.update(40)

        return df

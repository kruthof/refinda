from refinda.datasets.downloader import downloader
import pkg_resources
import pandas as pd
"""
Functions that download datasets from the cloud
"""


def simple_pre_processor(data):
    del data["Unnamed: 0"]
    return data


def french_smb_hml_mkt_rf():
    file = "https://dl.dropboxusercontent.com/s/ar3qvysnlveobfx/French_smb_hml_mkt_rf.csv.zip?dl=0"
    data = simple_pre_processor(downloader(file))
    return data


def french_mkt_size_bm_weighted_rf():
    file = "https://dl.dropboxusercontent.com/s/f1u94i2mdp0wznx/French_mkt_size_bm_weighted.csv.zip?dl=0"
    data = simple_pre_processor(downloader(file))
    return data


def french_mkt_size_bm_weighted_smb_hml_rf():
    file = "https://dl.dropboxusercontent.com/s/ohop3y6h1rc7pyn/French_mkt_size_bm_weighted_smb_hml.csv.zip?dl=0"
    data = simple_pre_processor(downloader(file))
    return data


def french_mkt_size_bm_weighted_smb_hml_mom_rf():
    file = "https://dl.dropboxusercontent.com/s/m54ehysgt52zv40/French_mkt_size_bm_weighted_smb_hml_mom.csv.zip?dl=0"
    data = simple_pre_processor(downloader(file))
    return data


def french_10ind_weighted_mkr_rf():
    file = "https://dl.dropboxusercontent.com/s/6mpuerlui6mo264/French_10_mkt_rf_weighted.csv.zip?dl=0"
    data = simple_pre_processor(downloader(file))
    return data

def french_10ind_weighted_mkr_rf():
    file = "https://dl.dropboxusercontent.com/s/6mpuerlui6mo264/French_10_mkt_rf_weighted.csv.zip?dl=0"
    data = simple_pre_processor(downloader(file))
    return data

def get_rf():
    file = "https://dl.dropboxusercontent.com/s/nqnsn8au27wy226/rf_rates.zip?dl=0"
    df = downloader(file)
    df['date'] = pd.to_datetime(df.date, format="%Y%m%d")
    df['date'] = [x.strftime("%Y-%m-%d") for x in df['date']]
    df.set_index('date', inplace=True)
    return df / 100  # convert to percentage

"""
def get_rf():
    '''
    Function loads rf rates, transforms date column and set it as index

    @return df dataframe rf rates
    '''
    df = pd.read_csv(pkg_resources.resource_stream(__name__, 'rf_rates.csv'))
    #df = pd.read_csv('./datasets/rf_rates.csv')
    df['date'] = pd.to_datetime(df.date, format="%Y%m%d")
    df['date'] = [x.strftime("%Y-%m-%d") for x in df['date']]
    df.set_index('date',inplace=True)
    return df/100 #convert to percentage
"""
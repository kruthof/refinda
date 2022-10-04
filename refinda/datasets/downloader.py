import warnings
from tqdm import tqdm
import pandas as pd


def downloader(file=None):
    warnings.filterwarnings("ignore")

    with tqdm(total=100) as pbar:

        # update progress bar
        pbar.update(10)

        data = pd.read_csv(file, compression="zip")

        # update progress bar
        pbar.update(100)

        return data

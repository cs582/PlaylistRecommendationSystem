import pandas as pd
import numpy as np
from tqdm import tqdm


class Preprocessing:
    def __init__(self, filename):
        self.filename = filename
        self.df_songs = None
        self.df_history = None

    def cleaning(self):
        """
        This method retrieves and cleans the dataframe before proceeding to the recommendation system.
        :return: pd.DataFrame
        """
        df_history = pd.read_csv(self.filename)
        df_history = df_history.drop_duplicates()

        return self.df
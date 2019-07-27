"""Includes functions to help normalize the data"""
import pandas as pd


class TitanicNormalizer:

    def __init__(self, titanic_df: pd.DataFrame) -> None:
        self.titanic_df = titanic_df

    def normalize(self):
        """
        Currently we do not normalize any feature in our ML pipeline
        :return:
        """
        pass

"""Includes functions to help clean the data"""
import pandas as pd
import numpy as np


class TitanicCleaner:

    def __init__(self, titanic_df: pd.DataFrame):
        self.titanic_df = titanic_df

    def display_head(self, rows: int = 5) -> pd.DataFrame:
        return self.titanic_df.head(rows)

    def remove_irrelavant_features(self) -> pd.DataFrame:
        """
        Removes the features PassengerId, Ticket number, Cabin number
        :return: Titanic dataframe
        """
        self.titanic_df.drop(labels=['PassengerId', 'Ticket', 'Cabin'], axis='columns', inplace=True)
        return self.titanic_df

    def extract_titles(self) -> pd.DataFrame:
        """
        (i)     Extracts titles from Name,
        (ii)    categorizes the titles
        (iii)   removes feature Name
        :return: Titanic Dataframe
        """
        self.titanic_df['Title'] = self.titanic_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        self.titanic_df.replace(to_replace=['Lady', 'Countess','Capt', 'Col',\
                                            'Don', 'Dr', 'Major', 'Rev', 'Sir', \
                                            'Jonkheer', 'Dona'], value='Rare', inplace=True)
        self.titanic_df.replace(to_replace='Mlle', value='Miss', inplace=True)
        self.titanic_df.replace(to_replace='Ms', value='Miss', inplace=True)
        self.titanic_df.replace(to_replace='Mme', value='Mrs', inplace=True)
        self.titanic_df.drop(labels=['Name'], inplace=True, axis='columns')
        return self.titanic_df

    def replace_null_embarked(self) -> pd.DataFrame:
        """
        Replaces null values in Embarked and Age
        We replace null values in Embarked with the mode (highest frequency of value)
        :return: Titanic Dataframe
        """
        embarked_mode = self.titanic_df['Embarked'].dropna().mode()
        self.titanic_df.fillna(value=embarked_mode)
        return self.titanic_df

    def replace_null_age(self, concat_df: pd.DataFrame) -> pd.DataFrame:
        """
        We guess the age based on the Title. We find median age of each Title group and assign that value
        :return: Titanic Dataframe
        """
        median_ages = concat_df[['Title', 'Age']].groupby(by='Title').median()

        def guess_age(title: str):
            if title is 'Master':
                return median_ages.loc['Master'].Age
            elif title is 'Mr':
                return median_ages.loc['Mr'].Age
            elif title is 'Mrs':
                return median_ages.loc['Mrs'].Age
            elif title is 'Miss':
                return median_ages.loc['Miss'].Age
            elif title is 'Rare':
                return median_ages.loc['Rare'].Age
            else:
                return median_ages.loc['Mr'].Age

        self.titanic_df = self.titanic_df.apply(lambda row: guess_age(row.Title) if np.nan(row.Age) else row.Title, axis=1)
        return self.titanic_df

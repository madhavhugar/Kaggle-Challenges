"""Includes functions to help clean the data"""
import pandas as pd


class TitanicCleaner:

    def __init__(self, titanic_df: pd.DataFrame):
        """
        Constructor for TitanicCleaner class
        :param titanic_df:
        """
        self.titanic_df = titanic_df

    def display_head(self, rows: int = 5) -> pd.DataFrame:
        """
        Displays the titanic_df Dataframe
        :param rows: number of rows to be displayed
        :return: Head of Titanic Dataframe
        """
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
        Replaces null values in Embarked
        We replace null values in Embarked with the mode (highest frequency of value)
        :return: Titanic Dataframe
        """
        embarked_mode = self.titanic_df['Embarked'].dropna().mode()
        self.titanic_df.fillna(value=embarked_mode)
        return self.titanic_df

    def replace_null_age(self) -> pd.DataFrame:
        """
        We impute the age based on the Title. We find median age of each Title group and assign that value
        :return: Titanic Dataframe
        """
        median_ages = self.titanic_df[['Title', 'Age']].groupby(by='Title').median()

        def guess_age(row: pd.DataFrame):
            if row['Title'] is 'Master':
                return median_ages.loc['Master'].Age
            elif row['Title'] is 'Mr':
                return median_ages.loc['Mr'].Age
            elif row['Title'] is 'Mrs':
                return median_ages.loc['Mrs'].Age
            elif row['Title'] is 'Miss':
                return median_ages.loc['Miss'].Age
            elif row['Title'] is 'Rare':
                return median_ages.loc['Rare'].Age
            else:
                return median_ages.loc['Mr'].Age

        self.titanic_df['GuessedAge'] = self.titanic_df.apply(guess_age, axis=1)
        self.titanic_df['Age'].fillna(self.titanic_df['GuessedAge'], inplace=True)
        self.titanic_df.drop(labels='GuessedAge', inplace=True, axis=1)
        return self.titanic_df

    def combine_family_features(self) -> pd.DataFrame:
        """
        We combine the number of parents/children feature and number of siblings
        :return: Titanic Dataframe
        """
        self.titanic_df['Family'] = self.titanic_df['Parch'] + self.titanic_df['SibSp']
        self.titanic_df.drop(labels=['Parch', 'SibSp'], inplace=True, axis=1)
        return self.titanic_df

    def get_dummies_categorical_features(self) -> pd.DataFrame:
        """
        We convert Sex, Title and Embarked into categorical features
        :return:
        """
        self.titanic_df['Sex'] = pd.Categorical(self.titanic_df['Sex'])
        self.titanic_df['Title'] = pd.Categorical(self.titanic_df['Title'])
        self.titanic_df['Embarked'] = pd.Categorical(self.titanic_df['Embarked'])
        self.titanic_df = pd.get_dummies(self.titanic_df, drop_first=True)
        return self.titanic_df

    def bin_age(self, age_group: list, group_names: list) -> pd.DataFrame:
        self.titanic_df['Age'] = pd.cut(self.titanic_df['Age'], age_group, labels=group_names)
        return self.titanic_df

    def bin_fare_pclass(self, fare_group: list, group_names: list) -> pd.DataFrame:
        df_fare_class = pd.cut((self.titanic_df['Fare'] * self.titanic_df['Pclass']),\
                                bins=fare_group, labels=group_names)
        self.titanic_df['Fare*Pclass'] = df_fare_class
        self.titanic_df.drop(labels=['Fare'], inplace=True, axis=1)
        return self.titanic_df

    def get_titanic_df(self) -> pd.DataFrame:
        """
        We return the Titanic Dataframe
        :return: Returns features and target
        """
        return self.titanic_df

    def get_feature_target(self) -> (pd.DataFrame, pd.Series):
        """
        We return features and target as a tuple
        :return: tuple(features Dataframe, target Dataframe)
        """
        df_features = self.titanic_df.loc[:, self.titanic_df.columns != 'Survived']
        df_target = self.titanic_df.loc[:, self.titanic_df.columns == 'Survived']
        return df_features, df_target

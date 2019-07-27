"""Includes functions to train and evaluate the model"""
import os
import pickle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def define_model(df_features, df_target):
    """
    Return the defined logistic regression model

    :param df_features: features training data
    :param df_target: target training data
    :return: Logistic Regressor model
    """
    log_regressor = linear_model.LogisticRegression()
    log_regressor.fit(df_features, df_target)
    return log_regressor


def export_model_pckl(ml_model, path):
    """
    Exports the machine learning model as a pickel file

    :param ml_model: machine learning model to be saved
    :param path: path to store the model
    :return: None
    """
    with open(os.path.join(path, 'model.pckl'), 'wb') as f:
        pickle.dump(ml_model, f)


def import_model_pckl(path):
    """
    Imports the machine learning mode from a pickel file

    :param path: path to the pickle file
    :return: machine learning model
    """
    with open(os.path.join(path, 'model.pckl'), 'rb') as f:
        return pickle.load(f)


def predict_target(ml_model, df_features):
    """
    Predicts the survival of the Titanic passengers

    :param ml_model: The machine learning model
    :param df_features: the input features to be predicted for
    :return: Prediction pandas series
    """

    return ml_model.predict(df_features)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from datetime import datetime
# from gluonts.dataset.pandas import PandasDataset
# from gluonts.dataset.split import split
# from gluonts.mx import DeepAREstimator, Trainer #, TrainerCallback, EarlyStoppingCallback
from gluonts.dataset.common import ListDataset


def format(df):
  # this function prepares the data for Deep AR
  df = df.set_index('Date')
  return df

def data_prep(df, ticker, dyn_feat, horizon):

    # this function prepares the data to feed into DeepAR
    # df: Data Frame
    # Ticker: target Ticker
    # dny_feat: List of covariates
    # horizon: the forecast length

    train_data = ListDataset(
        [
            {"start": df.index[0],
             "target": df[df['ticker']==str(ticker)]['target'].values,
             "dynamic_feat": dyn_feat}
        ],
        freq="M"
    )

    test_data = ListDataset(
        [
            {"start": df.index[-horizon],
             "target": df[df['ticker']==str(ticker)]['target'][-12:].values,
             "dynamic_feat": dyn_feat}
        ],
        freq="M"
    )
    return train_data, test_data


def fit_n_predict(model, training, test):
  
  # this function fist the model and one can check the forecast horizon

  predictor = model.train(training)
  predictor = next(predictor.predict(test))
  samples = predictor.samples
  time = predictor.index
  return samples, time


def tick_loss(alpha, returns, var):
    df = pd.DataFrame({'Return': returns, 'VaR': var})
    df['Indicator'] = np.where(df['Return'] < df['VaR'], 1, 0)

    t_loss = 0

    for i in df.index:
        t_loss += (alpha * (df['Return'][i] - df['VaR'][i]) * (1 - df['Indicator'][i]) +
                 (1 - alpha) * abs((df['Return'][i] - df['VaR'][i]))* df['Indicator'][i])

    return t_loss



import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV

def tick_loss(alpha, returns, var):
    df = pd.DataFrame({'Return': returns, 'VaR': var})
    df['Indicator'] = np.where(df['Return'] < df['VaR'], 1, 0)
    
    t_loss = 0

    for i in df.index:
        t_loss += (alpha * (df['Return'][i] - df['VaR'][i]) * (1 - df['Indicator'][i]) + 
                 (1 - alpha) * abs((df['VaR'][i] - df['Return'][i])) * df['Indicator'][i])

    return t_loss



def get_historical_quantiles(df, column_names, alpha):


  percentile_1 = df[column_names].quantile(alpha)
 

  return percentile_1


def best_model(model, X, y):
   

   def tick_loss(alpha, returns, var):
    df = pd.DataFrame({'Return': returns, 'VaR': var})
    df['Indicator'] = np.where(df['Return'] < df['VaR'], 1, 0)
    
    t_loss = 0

    for i in df.index:
        t_loss += (alpha * (df['Return'][i] - df['VaR'][i]) * (1 - df['Indicator'][i]) + 
                 (1 - alpha) * abs((df['VaR'][i] - df['Return'][i])) * df['Indicator'][i])

    return t_loss
   
   refcv = RFECV(estimator=model, scoring=tick_loss, cv=5, verbose=1, n_jobs=4)

   refcv.fit(X, y)

   refcv.transform(X)

   print(); print(refcv)
   print(); print(refcv.n_features_)

   
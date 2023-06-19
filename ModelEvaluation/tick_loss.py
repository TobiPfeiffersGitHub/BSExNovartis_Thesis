import pandas as pd
import numpy as np

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
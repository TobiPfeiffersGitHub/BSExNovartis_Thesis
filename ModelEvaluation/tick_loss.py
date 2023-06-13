import pandas as pd
import numpy as np

def tick_loss(alpha, returns, var):
    df = pd.DataFrame([returns, var], columns=['Return', 'VaR'])
    df['Indicator'] = np.where(df['Return'] < df['VaR'], 1, 0)
    
    t_loss = 0

    for i in df.index:
        t_loss += (alpha * (df['Return'][i] - df['VaR'][i]) * (1 - df['Indicator'][i]) + 
                 (1 - alpha) * (df['VaR'][i] - df['Return'][i]) * df['Indicator'][i])
    print(df)
    return t_loss



def get_historical_quantiles(df, column_names, alpha):
  """
  Args:
    df: A pandas DataFrame.
    column_names: A list of column names.

  Returns:
    A pandas DataFrame with the column names and the 5th percentile.
  """

  percentile_1 = df[column_names].quantile(alpha)
 #   percentile_5 = df[column_names].quantile(0.05)
 #   percentile_10 = df[column_names].quantile(0.10)

 #   percentile_table = pd.DataFrame({
 #     "1st Percentile": percentile_1,
 #     "5th Percentile": percentile_5,
 #     "10th Percentile": percentile_10
 #   })

  return percentile_1
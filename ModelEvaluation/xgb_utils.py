import numpy as np
np.random.seed(1)
import pandas as pd
import scipy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns


def collect_prediction(X_train,y_train,X_test,y_test,estimator,alpha,model_name):
  estimator.fit(X_train,y_train)
  y_pred = estimator.predict(X_test)
  #print( "{model_name} alpha = {alpha:.2f},score = {score:.1f}".format(model_name=model_name, alpha=alpha , score= XGBQuantile.quantile_score(y_test, y_pred, alpha)) )

  return y_pred

class XGBQuantile(XGBRegressor):
  def __init__(self,quant_alpha=0.95,quant_delta = 1.0,quant_thres=1.0,quant_var =1.0,base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                n_jobs=1, nthread=None, objective='reg:linear', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,silent=True, subsample=1):
    self.quant_alpha = quant_alpha
    self.quant_delta = quant_delta
    self.quant_thres = quant_thres
    self.quant_var = quant_var

    super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
       colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, max_delta_step=max_delta_step,
       max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators,
       n_jobs= n_jobs, nthread=nthread, objective=objective, random_state=random_state,
       reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed,
       silent=silent, subsample=subsample)

    self.test = None

  def fit(self, X, y):
    super().set_params(objective=partial(XGBQuantile.quantile_loss,alpha = self.quant_alpha,delta = self.quant_delta,threshold = self.quant_thres,var = self.quant_var) )
    super().fit(X,y)
    return self

  def predict(self,X):
    return super().predict(X)

  def score(self, X, y):
    y_pred = super().predict(X)
    score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
    score = 1./score
    return score

  @staticmethod
  def quantile_loss(y_true,y_pred,alpha,delta,threshold,var):
    x = y_true - y_pred
    grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-  ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
    hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta

    grad = (np.abs(x)<threshold )*grad - (np.abs(x)>=threshold )*(2*np.random.randint(2, size=len(y_true)) -1.0)*var
    hess = (np.abs(x)<threshold )*hess + (np.abs(x)>=threshold )
    return grad, hess

  @staticmethod
  def original_quantile_loss(y_true,y_pred,alpha,delta):
    x = y_true - y_pred
    grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
    hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta
    return grad,hess


  @staticmethod
  def quantile_score(y_true, y_pred, alpha):
    score = XGBQuantile.quantile_cost(x=y_true-y_pred,alpha=alpha)
    score = np.sum(score)
    return score

  @staticmethod
  def quantile_cost(x, alpha):
    return (alpha-1.0)*x*(x<0)+alpha*x*(x>=0)

  @staticmethod
  def get_split_gain(gradient,hessian,l=1):
    split_gain = list()
    for i in range(gradient.shape[0]):
      split_gain.append(np.sum(gradient[:i])**2/(np.sum(hessian[:i])+l)+np.sum(gradient[i:])**2/(np.sum(hessian[i:])+l)-np.sum(gradient)**2/(np.sum(hessian)+l) )

    return np.array(split_gain)


def tick_loss(alpha, returns, var):
    df = pd.DataFrame({'Return': returns, 'VaR': var})
    df['Indicator'] = np.where(df['Return'] < df['VaR'], 1, 0)

    t_loss = 0
    for i in df.index:
        t_loss += (alpha * (df['Return'][i] - df['VaR'][i]) * (1 - df['Indicator'][i]) +
                 (1 - alpha) * abs((df['VaR'][i] - df['Return'][i])) * df['Indicator'][i])

    return t_loss


def plot_feats(fitted_model, X):
    fitted_model.feature_importances_

    features = X.columns

    importances_clf = pd.Series(data = fitted_model.feature_importances_,
                                index= features)

    # Sort importances
    importances_sorted_clf = importances_clf.sort_values()

    # Draw a horizontal barplot of importances_sorted
    importances_sorted_clf.plot(kind='barh', color='sandybrown')
    plt.title('Gini Features Importances', size=15)
    plt.show()


def get_model_performance(data, ticker, alpha, exclude, dates, horizon, params, features = False, gcv=False):
    """
    Args: data = data, ticker = ticker, alpha = risk, dates = string to split the dataframe, horizong,
    e.g., 3m, 9m, etc.. input just the number,  
    """

    t_loss = 0
    dta = data.copy()

    for d in dates:
        split_date = d
        train_df = dta[dta.Date <= split_date]
        train_df = train_df.dropna() 
        test_df = dta[dta.Date > split_date]
        test_df = test_df.reset_index(drop=True)

        X_train = train_df.drop(columns=exclude, axis=1)
        X_test = test_df.drop(columns=exclude, axis=1)
        X_test = X_test.iloc[0:horizon]
        y_train = train_df[ticker]
        y_test = test_df[ticker][0:horizon]

        param_grid = params

        regressor = GradientBoostingRegressor()

        # Perform grid search cross-validation
        grid_search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error', cv=3)
        grid_search.fit(X_train, y_train)

        best_regressor = grid_search.best_estimator_

        y_pred = best_regressor.predict(X_test)

        regressor.set_params(loss='quantile', alpha=0.05)
        y_lower = collect_prediction(X_train, y_train, X_test, y_test, estimator=best_regressor, alpha=0.05, model_name="Gradient Boosting")

        regressor.set_params(loss='quantile', alpha=0.95)
        y_upper = collect_prediction(X_train, y_train, X_test, y_test, estimator=best_regressor, alpha=0.95, model_name="Gradient Boosting")

        tickloss = tick_loss(alpha, y_test, y_lower)
        #print(tickloss)

        if features == True:
           plot_feats(best_regressor, X_train)

        t_loss += tickloss

    combinations = 13-horizon
    t_loss = t_loss / combinations
    return t_loss


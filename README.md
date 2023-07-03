# BSE Master Thesis
(In cooperation with the Novartis Digital Finance Hub)

#### Authors: Lluïsa Rull Ferran, Maria Pintos Relat, Tobias Pfeiffer

### Statemnet of Purpose:
For our Master Thesis we partnered with the Novartis Digital Finance Hub to to model the downside risk of 10 pharma stocks. Our paper/project compares models from different field to investigate strength and weakness of models, and tries to find the best model for each stock in the portfolio.


### Structure of this Repository
Our repository consists mainly out of 3 folders:
- The *Data*-folder contains scripts to scrape data from the yahoo-finance library and the Fred-API. The data obtained from these processes is used throughout the repository.
- The *Exploratory Data Analyis*-folder contains scripts that perform some analysis like descriptive statistics, stationarity test, or valatility analysis.
- Finally, the *Model Evaluation* Folder contains all script where we modeled Value-at-Risk. We use six different models: Historical Quantile, GARCH, Quantile Regression, XGBoost quantile regression, DeepAR (a neural network architecture for time series), and an ensemble model that takes the previous models into account.


### Data
All our data is on a monthly basis and ranges from 01/2000 to 05/2023. Appart from the stock return which we calculate from the yahoo finance data we also include macroeconomic variables. The variables are (change of) inflation, (change of) unemployment, (change of) treasury yield, GDP growth, and S&P500 returns. Moreover, we include dummy variables to absorb effects of earnings calls and macro-data releases.


### Methodology
Our first step was to define an evaluation metrics, in this case a tick-loss function which is defined as follows:
```math
TL = \sum_{t = T+1}^{T+h} \alpha |(r_t - \hat{v}_{t})| (1 - 1\{r_t < \hat{v}_{t}\})+(1-\alpha)|(r_t - \hat{v}_{t})|1\{r_t < \hat{v}_{t}\}
```
this function penelises "violations" of the forecast more than inefficent forecasts.

Braodly there are two ways to estimate VaR (which is esentially a quantile of the return distribution): uncoditional quantiles and conditional quantiles (conditional on covariates). We use the historical quantile (basically an order statistic) as a baseline. We then tried various models to improve upon the baseline:
- **GARCH**: By estimating the conditional variance of the data series, GARCH models allow for the identification of time-varying volatility patterns, which are crucial for predicting future risks. It incorporates both autoregressive and moving average components to capture the persistence and volatility clustering observed in financial data.
- **Quantile Regression (QR)**: QR estimates quantiles of a variable. Since Value-at-Risk can be viewed as a quantile, QR allows us to model Value-at-Risk conditional on the covariates. It works by minimising the aforementioned tick-loss function.
- **XGBoost**: XGBoost is an alorithm that chains many descion trees together to make predictions. However,  





 

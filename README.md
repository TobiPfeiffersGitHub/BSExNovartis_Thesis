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
- **XGBoost**: XGBoost is an alorithm that chains many descion trees together to make predictions. However, to get an estimate for Value-at-Risk we needed to alther the alogrithm and impose the tickloss as the cost function.
- **DeepAR**: DeepAR is a neural network architecture, designed for time series. It uses a neural network to parametrize a likelihood function. From this distribution it draws $n$ samples. We use the 5th percentile of this sample as an estimate for the  Value-at-risk.
- **Ensemble**: We use an ensemble that upweights the tickloss of each model and returns a convex combination of the prediction vector from each model (excluding DeepAR).


### Results
None of the models consistently outperforms the baseline. However, some models perform well for some specifications:
- **Garch**: Lower loss values indicate a closer match between the model's predicted volatility and the observed volatility. Among the companies mentioned, NVS, BMY, JNJ, MRK, NVO, and ABBV demonstrate a better fit to the volatility patterns, as indicated by lower loss values. On the other hand, ROG consistently shows higher loss values, suggesting that the model struggles to accurately capture its volatility patterns. AZN, LLY, and PFE have moderate loss values, indicating some challenges in capturing their volatility patterns, but not as pronounced as with ROG.
- **Quantile Regression**: The analysis reveals that the performance varies across different tickers. AZN stands out as a relatively strong performer, particularly in shorter and medium-term time horizons. The overall model performs better than the baseline, but surprisingly, a naive model that only considers macroeconomic variables performs better for JNJ. Additionally, the Quantile Regression approach demonstrates superior performance in the short and medium-term compared to the long-term. These findings may be attributed to overfitting and multicollinearity issues. To enhance performance and mitigate multicollinearity problems, it may be beneficial to focus more on macroeconomic variables in the analysis.
- **XGBoost**: The model shows strong performance in the short term and is particularly effective when applied to low volatility stocks (such as JNJ, BMY, and MRK). Furthermore, the results are closely aligned with the historical baseline, indicating a reliable and consistent performance.
- **DeepAR**: DeepAR is one of the worse performing models. Its performance improves over longer time horizons, but is primarily driven by individual stock behavior rather than broader trends or model inputs. Table 10 shows the tick-losses of the Neural Network, with Merk performing well in the medium term across all specifications and outperforming the baseline. Johnson & Johnson also performs well in longer time horizons and is responsive to macroeconomic variables. The underperformance of DeepAR may be attributed to limited data availability, as the model is trained on a relatively small dataset of 280 to 300 observations. To address this, utilizing more granular data such as daily or hourly data could be explored, although this would require compromising on the inclusion of macro specifications.


### Final Thoughts
Our thesis showed a view interesting results: Unconditional methods work better to model risk. The performance of the conditional methods is bound to the volatility of the series. It would be interesting how the machine learning models fare with more data (e.g., use daily data) and tuned hyper parameters. 

 

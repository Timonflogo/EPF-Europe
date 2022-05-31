# EPF-Europe
Electricity prices have become increasingly volatile within the recent past
due to an increasing influx of intermittent energy production through re-
newables, green energy goals of governments and enterprises, and increasing
demand for the further development of electrification infrastructures. Fur-
thermore, economic shocks such as the Covid-19 pandemic which already put
Europe in an unfavourable position in regards to their energy resources, the
recent developments in the Ukrainian-Russian conflict further destabilise Eu-
rope’s energy supply. As a result, electricity prices have been soaring and elec-
tricity market volatility is around five times larger over the period 2021/2022,
than in the years prior. Moreover, it has become increasingly more difficult
to obtain accurate forecasts of day-ahead electricity prices, further raising un-
certainty for stakeholders such as governments, power traders, transmission
system operators, manufacturing companies, and households.
Increasingly, machine and deep learning methods find their way into the
realm of time series forecasting of electricity prices. This Thesis builds on
top of the current state of the literature, by investigating the short-term fore-
casting performance of non-linear state-of-the-art Electricity Price Forecasting
methods XGBoost, Support Vector Regressor, Long-Short-Term-Memory, and
Gated Recurrent Unit Neural Networks in periods of high electricity market
volatility. The data used in this thesis are six univariate electricity price se-
ries of hourly frequency, from different bidding areas of the Nordpool market.
Furthermore, the effect of different levels of renewable energy on forecast-
ing performance is investigated by using Germany, Denmark, Norway, and
Sweden.
Different volatility levels have been detected for the period of march 2017
to march 2022 . The series have been found to be stationary but entering
a period of high volatility and non-stationary behaviour since early 2021.
In addition to XGBoost, SVR, LSTM, and GRU forecasters,three statistical
baseline models AR, ARIMA, and SARIMA were deployed as a benchmark.
In total 42 short-term forecasts of 24 observations were obtained.
It was found that overall, all non-linear machine and deep learning models
outperformed the statistical baseline models. On average, the lowest error on
short-term forecasts of 24 hourly observations is a Neural Network featuring
a GRU layer of 64 neurons and a Dense layer of 8 neurons, followed by XG-
Boost, LSTM, and SVR. The least erroneous forecast for the most volatile
test set, from the bidding area SE4, was obtained by a regularised Support
Vector Regressor using a Radial Basis Function. Morover, XGBoost took the
longest to train and obtain forecasts with computation times taking around
500 times longer than a simple AR, followed by LSTM and GRU with compu-
tation times taking around 350 times longer than a simple AR. Furthermore,
Forecasting performance could be interpreted to be more accurate in countries
with lower levels of renewable energy consumption, However, the results are
inconclusive.

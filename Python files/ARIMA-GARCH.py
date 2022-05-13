#%% import libraries
import pandas as pd
import numpy as np

#%% import data
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
DE = HMV[['DE']]

#%% # Create the windowed dataset
windowLength = 500
foreLength = len(DE['DE']) - windowLength

windowed_ds = []
for d in range(foreLength-1):
    windowed_ds.append(DE['DE'][d:d + windowLength])

# create the forecasts dataframe initialized with zeros
forecasts = DE['DE'].iloc[windowLength:].copy() * 0

windowed_ds[0], forecasts, windowed_ds[-1]

#%% create fit_arima function to receive a series and return the best fit
import pmdarima
import arch

import warnings
warnings.filterwarnings("ignore")

def fit_arima(series, range_p=range(0, 6), range_q=range(0, 6)):
    final_order = (0, 0, 0)
    best_aic = np.inf
    arima = pmdarima.ARIMA(order=final_order)

    for p in range_p:
        for q in range_q:
            if (p==0) and (q==0):
                next
            arima.order = (p, 0, q)
            arima.fit(series)

            aic = arima.aic()

            if aic < best_aic:
                best_aic = aic
                final_order = (p, 0, q)
                
    arima.order=final_order
    return arima.fit(series)

# create loop to forecast both models by 1 period

for i, window in enumerate(windowed_ds):
    # ARIMA model
    arima = fit_arima(window)
    arima_pred = arima.predict(n_periods=1)
    
    # GARCH model
    garch = arch.arch_model(arima.resid())
    garch_fit = garch.fit(disp='off', show_warning=False, )
    garch_pred = garch_fit.forecast(horizon=1).mean.iloc[-1]['h.1']    
    
    forecasts.iloc[i] = arima_pred + garch_pred
    
    print(f'Date {str(forecasts.index[i].date())} : Fitted ARIMA order {arima.order} - Prediction={forecasts.iloc[i]}')
    
#%% First, we will save our newly created signals
forecasts.to_csv('new_python_forecasts.csv')

# Get the period of interest
forecasts = forecasts[(forecasts.index>='2002-01-01') & (forecasts.index<='2020-12-31')]

# Get the direction of the predictions
forecasts['Signal'] = np.sign(forecasts['Signal'])

DE.add_signal_strategy(forecasts, column_name='Signal')

df = DE.compare_strategy(start='2002-01-02', end='2010-12-01', figsize=(15,7))
plt.ylabel('Strategies Return (%)')
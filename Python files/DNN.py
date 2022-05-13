#%% import libraries
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import seaborn as sns

#%% Setup plotting environment
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=1.8)

# set plotting parameters
rcParams['figure.figsize'] = 20, 12  
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=4, linestyle='-')
# rcParams['text.usetex'] = True

#%% import dataset
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
DE = HMV['DE']

#%% plot data
DE.plot()

#%% create windowing function
def df_to_supervised(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

#%% use df_to_supervised function
WINDOW_SIZE = 336
X, y = df_to_supervised(DE, WINDOW_SIZE)
print(X.shape, y.shape)

#%% create train, val, and test set
val_horizon = 168
test_horizon = 24

X_train, y_train = X[:-(val_horizon + test_horizon)], y[:-(val_horizon + test_horizon)]
X_val, y_val = X[len(X_train): (len(X_train) + val_horizon)], y[len(X_train): (len(X_train) + val_horizon)]
X_test, y_test = X[(len(X_train) + val_horizon):], y[(len(X_train) + val_horizon):]
    
# get shapes
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

#%% import libraries for Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

#%% BUILD DNN
model_DNN = Sequential()
model_DNN.add(InputLayer((X_train.shape[1], X_train.shape[2])))
model_DNN.add(LSTM(16))
model_DNN.add(Dropout(0.5))
model_DNN.add(Dense(8, activation = 'relu'))
model_DNN.add(Dropout(0.5))
model_DNN.add(Dense(1, activation = 'linear'))

model_DNN.summary()

#%% callbacks
cp = ModelCheckpoint('model_DNN/', save_best_only=True)

#%% compile
model_DNN.compile(loss='mse',
               optimizer=Adam(learning_rate=0.01),
               metrics=[MeanAbsoluteError()])

#%% fit model
history = model_DNN.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=10,
                     callbacks=[cp])

#%% plot history loss 
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#%% load the model
from tensorflow.keras.models import load_model
model_DNN = load_model('model_DNN/')

#%% prediction Train and plot train results
train_predictions = model_DNN.predict(X_train)
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
print(train_results)

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][-100:])
plt.plot(train_results['Actuals'][-100:])

#%% predict Val and plot val results
val_predictions = model_DNN.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
print(val_results)

plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

#%% predict test and plot test results
test_predictions = model_DNN.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
print(test_results)

plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])

#%% get error metrics 
from sklearn.metrics import mean_absolute_percentage_error as mape
error_LSTM_DE = mape(test_results['Actuals'], test_results['Test Predictions'])*100
print(error_LSTM_DE)


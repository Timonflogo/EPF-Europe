#%% import libraries
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

#%% Create Class for colour coding of ADF test results to enhance readability in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[36m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

#%% import libraries for LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

#%% prepare data and isolate series
df = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True)
DE = df[["DE"]]
DK1 = df[["DK1"]]
DK2 = df[["DK2"]]
SE3 = df[["SE3"]]
SE4 = df[["SE4"]]
NO2 = df[["NO2"]]

#%% define window length and forecasting horizon
n_input = 168
horizon = 24

###################################################################################
#%% train test split for DE
# Train Test set split - we want to forecast 1 month into the future so out test set should be at least one month 
len(DE)
# we will go with a 90-10 train-test split such that our test set represents 3 months worth of data
trainDE =  DE[:len(DE)-horizon]
test_DE = DE[len(DE)-horizon:]
len(DE) == len(trainDE) + len(test_DE)

#%% Scale data
scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(trainDE)
scaled_train = scaler.transform(trainDE)
scaled_test = scaler.transform(test_DE)
# scaled_train = train
# scaled_test = test

#%% time series generator 
# Let's define to get 168 Days back wich represents one week and then predict the next week out
n_input = n_input
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=10)

# Check Generated time series object 
len(scaled_train)
len(generator) # n_input = 2
scaled_train
X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')

#%% DEFINE THE MODEL 
# define model
model = Sequential()
model.add(LSTM(100, activation='tanh', 
               input_shape=(n_input, n_features),
               dropout=0.1))
model.add(Dense(1))
# comile model
model.compile(optimizer='adam', loss='mse')

#%% get model summary
model.summary()

#%% Callback early stopping
callback = EarlyStopping(monitor='loss', patience=5)

#%% fit model
import time
start_time = time.time()
model.fit_generator(generator,epochs=100,callbacks=callback,shuffle=False)
LSTM_DE_time = (time.time() - start_time)

 #%% model performance
model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# Evaluate on Test Data
# first_eval_batch = scaled_train[-24:]
# first_eval_batch
# first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
# model.predict(first_eval_batch)
# scaled_test[0]

#%% LOOP to get predictions for entire test set
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test_DE)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
test_predictions

#%% INverse transform to compare to actual data
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions

#%% Append predictions to test set
test_DE['LSTM'] = true_predictions

#%% plot predictions 
plt.plot(test_DE['DE'], label="DE")
plt.plot(test_DE['LSTM'], linestyle="--", label="LSTM", color='red')
plt.legend()
plt.show()

#%% Save test data and prediction to csv
test_DE.to_csv()

#%% get error measures

LSTM_DE_error = mape(test_DE['DE'], test_DE['LSTM']) 


###################################################################################
#%% train test split for DK1
# Train Test set split - we want to forecast 1 month into the future so out test set should be at least one month 
len(DK1)
# we will go with a 90-10 train-test split such that our test set represents 3 months worth of data
train_DK1 =  DK1[:len(DK1)-horizon]
test_DK1 = DK1[len(DK1)-horizon:]
len(DK1) == len(train_DK1) + len(test_DK1)

#%% Scale data
scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train_DK1)
scaled_train = scaler.transform(train_DK1)
scaled_test = scaler.transform(test_DK1)
# scaled_train = train
# scaled_test = test

#%% time series generator 
# Let's define to get 168 Days back wich represents one week and then predict the next week out
n_input = n_input
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=10)

# Check Generated time series object 
len(scaled_train)
len(generator) # n_input = 2
scaled_train
X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')

#%% DEFINE THE MODEL 
# define model
model = Sequential()
model.add(LSTM(16, activation='tanh', input_shape=(n_input, n_features)))
model.add(Dense(1))
# comile model
model.compile(optimizer='adam', loss='mse')

#%% get model summary
model.summary()

#%% Callback early stopping
callback = EarlyStopping(monitor='loss', patience=5)

#%% fit model
import time
start_time = time.time()
model.fit_generator(generator,epochs=100,callbacks=callback,shuffle=False)
LSTM_DK1_time = (time.time() - start_time)

#%% model performance
model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# Evaluate on Test Data
# first_eval_batch = scaled_train[-24:]
# first_eval_batch
# first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
# model.predict(first_eval_batch)
# scaled_test[0]

#%% LOOP to get predictions for entire test set
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test_DK1)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
test_predictions

#%% INverse transform to compare to actual data
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions

#%% Append predictions to test set
test_DK1['DK1 LSTM Predictions'] = true_predictions

#%% plot predictions 
plt.plot(test_DK1['DE'], label="DK1")
plt.plot(test_DK1['LSTM'], linestyle="--", label="LSTM")
plt.legend()
plt.show()

#%% get error measures 
error = mape(test_DK1['DE'], test_DK1['Predictions'])


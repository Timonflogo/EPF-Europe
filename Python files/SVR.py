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
sns.set(style='whitegrid', palette='deep', font_scale=1)

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
    
#%% import libraries for SVR
import sys
sys.path.append('../../')
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape

#%% import dataset
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
DE = HMV[['DE']]

#%% define forecasting horizon
horizon = 24

#%% train test split
# we will go with a train-test split such that our test set represents 168 Hours worth of data
train =  DE[:len(DE)-horizon]
test = DE[len(DE)-horizon:]
len(DE) == len(train) + len(test) # True

# forecast start and end
# obtain predicted results
start = len(train)
end = len(train)+len(test)-1

#%% print shapes
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

#%% scale data
scaler = MinMaxScaler()
train['DE'] = scaler.fit_transform(train)

test['DE'] = scaler.transform(test)

#%% Create data with time-steps

# Converting to numpy arrays
train_data = train.values
test_data = test.values

# determine number of timesteps. If 5 then inputs will be 4 and output will be the 5th timestep
timesteps=5

# Converting training data to 2D tensor using nested list comprehension:
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
print("train data shape",train_data_timesteps.shape)

# Converting testing data to 2D tensor:
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
print("test data shape", test_data_timesteps.shape)

#%% Selecting inputs and outputs from training and testing data:
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Implement SVR 
model = SVR(kernel='rbf',
            gamma=0.5,
            C=10,
            epsilon = 0.05,
            degree=3,
            shrinking=True)

# fit the model on training data
print("running: ", model.fit(x_train, y_train[:,0]))

#%% make model predictions
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)

#%% Evaluate model 
train_timestamps = train.index[timesteps-1:]
test_timestamps = test.index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))

#%% Plot the predictions for training data
plt.figure(figsize=(25,10))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

#%% Get MAPE
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')

#%% plot predicitons for testing data
plt.figure(figsize=(12,6))
plt.plot(test_timestamps, y_test, color = 'blue', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'red', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

# get MAPE for test data
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
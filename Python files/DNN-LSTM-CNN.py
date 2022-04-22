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

mpl.rcParams['figure.figsize'] = (8, 6)
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
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

#%% prepare data
df = read_csv('NP-LMV.csv', index_col='HourDK', parse_dates=True)
df = df["DE"]

# Train Test set split - we want to forecast 1 month into the future so out test set should be at least one month 
len(df)
# we will go with a 90-10 train-test split such that our test set represents 3 months worth of data
train =  df[:len(df)-168]
test = df[len(df)-168:]
len(df) == len(train) + len(test)

# Scale data
scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
# scaled_train = train
# scaled_test = test


# Let's define to get 168 Days back wbich represents one week and then predict the next week out
n_input = 168
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=10)

# Check Generated time series object 
len(scaled_train)
len(generator) # n_input = 2
scaled_train
X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')

# DEFINE THE MODEL 
# define model
model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
# comile model
model.compile(optimizer='adam', loss='mse')

# get model summary
model.summary()

# fit model
model.fit_generator(generator,epochs=10, shuffle=False)

# model performance
model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# Evaluate on Test Data
# first_eval_batch = scaled_train[-24:]
# first_eval_batch
# first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
# model.predict(first_eval_batch)
# scaled_test[0]

# LOOP to get predictions for entire test set
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
test_predictions

# INverse transform to compare to actual data
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions

# IGNORE WARNINGS
test['Predictions'] = true_predictions

# plot predictions 
test.plot(figsize=(12,8))


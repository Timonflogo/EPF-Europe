#%% import libraries
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from matplotlib import pyplot

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

#%% Import dataset
LMV = read_csv('NP-LMV.csv', index_col='HourDK', parse_dates=True)
HMV = read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 

#%% XGBoost check
import xgboost
print("xgboost", xgboost.__version__)

#%% replace DateTime index with numbers
#LMV = LMV.reset_index()
#LMV = LMV.iloc[1:]

#%% Create dataframes for each LMV series
for i in LMV.columns[0:]:
    text=f"LMV_{i}=pd.DataFrame(LMV[i])"
    exec(text)

#%% Create dataframes for each HMV series
for i in LMV.columns[0:]:
    text=f"HMV_{i}=pd.DataFrame(HMV[i])"
    exec(text)

#%% create list of dataframes to do transformations on all variables. 
d = {}
for name in LMV.columns[0:]:
    d[name] = LMV[name]

#%% import libraries 
    
#%% create series_to_supervised function to enable supervised training for XGBoost
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

#%% define train test split function for univariate time series
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

#%% create xgboost forecaster
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(asarray([testX]))
	return yhat[0]

#%% Create autoregressive forecasting loop
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions
    
#%% create arrays with lags for LMV type datasets for XGBoost input
for i in LMV.columns[0:]:
    text=f"LMV_{i}=pd.DataFrame(series_to_supervised(LMV[[i]], n_in=24))"
    exec(text)

#%% create arrays with lags for LMV type datasets for XGBoost input
for i in HMV.columns[0:]:
    text=f"HMV_{i}=pd.DataFrame(series_to_supervised(HMV[[i]], n_in=24))"
    exec(text)

#%% create train and test sets of type LMV for XGBoost 
LMV_DEtrainX, LMV_DEtrainy = LMV_DE[:len(LMV_DE)-   168], LMV_DE[len(LMV_DE)-168:]
LMV_DK1trainX, LMV_DK1trainy = LMV_DK1[:len(LMV_DK1)-168], LMV_DK1[len(LMV_DK1)-168:]
LMV_DK2trainX, LMV_DK2trainy = LMV_DK2[:len(LMV_DK2)-168], LMV_DK2[len(LMV_DK2)-168:]
LMV_NO2trainX, LMV_NO2trainy = LMV_NO2[:len(LMV_NO2)-168], LMV_NO2[len(LMV_NO2)-168:]
LMV_SE3trainX, LMV_SE3trainy = LMV_SE3[:len(LMV_SE3)-168], LMV_SE3[len(LMV_SE3)-168:]
LMV_SE4trainX, LMV_SE4trainy = LMV_SE4[:len(LMV_SE4)-168], LMV_SE4[len(LMV_SE4)-168:]

#%% create train and test sets of type HMV for XGBoost 
LMV_DEtrainX, LMV_DEtrainy = LMV_DE[:len(LMV_DE)-168], LMV_DE[len(LMV_DE)-168:]
LMV_DK1trainX, LMV_DK1trainy = LMV_DK1[:len(LMV_DK1)-168], LMV_DK1[len(LMV_DK1)-168:]
LMV_DK2trainX, LMV_DK2trainy = LMV_DK2[:len(LMV_DK2)-168], LMV_DK2[len(LMV_DK2)-168:]
LMV_NO2trainX, LMV_NO2trainy = LMV_NO2[:len(LMV_NO2)-168], LMV_NO2[len(LMV_NO2)-168:]
LMV_SE3trainX, LMV_SE3trainy = LMV_SE3[:len(LMV_SE3)-168], LMV_SE3[len(LMV_SE3)-168:]
LMV_SE4trainX, LMV_SE4trainy = LMV_SE4[:len(LMV_SE4)-168], LMV_SE4[len(LMV_SE4)-168:]

#%% Run XGBoost on LMV_DE data
values_LMV_DE = LMV[['DE']].values
# transform the time series data into supervised learning
data_LMV_DE = series_to_supervised(values_LMV_DE, n_in=24)
# evaluate
import time
start_time = time.time()
mae_LMV_DE, y_LMV_DE, yhat_LMV_DE = walk_forward_validation(data_LMV_DE, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_LMV_DE)
# plot expected vs preducted
pyplot.plot(y_LMV_DE, label='Expected')
pyplot.plot(yhat_LMV_DE, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on HMV_DE
values_HMV_DE = HMV[['DE']].values
# transform the time series data into supervised learning
data_HMV_DE = series_to_supervised(values_HMV_DE, n_in=24)
# evaluate
import time
start_time = time.time()
mae_HMV_DE, y_HMV_DE, yhat_HMV_DE = walk_forward_validation(data_HMV_DE, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_HMV_DE)
# plot expected vs preducted
pyplot.plot(y_HMV_DE, label='Expected')
pyplot.plot(yhat_HMV_DE, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on LMV_DK1 data
values_LMV_DK1 = LMV[['DK1']].values
# transform the time series data into supervised learning
data_LMV_DK1 = series_to_supervised(values_LMV_DK1, n_in=24)
# evaluate
import time
start_time = time.time()
mae_LMV_DK1, y_LMV_DK1, yhat_LMV_DK1 = walk_forward_validation(data_LMV_DK1, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_LMV_DK1)
# plot expected vs preducted
pyplot.plot(y_LMV_DK1, label='Expected')
pyplot.plot(yhat_LMV_DK1, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on HMV_DK1
values_HMV_DK1 = HMV[['DK1']].values
# transform the time series data into supervised learning
data_HMV_DK1 = series_to_supervised(values_HMV_DK1, n_in=24)
# evaluate
import time
start_time = time.time()
mae_HMV_DK1, y_HMV_DK1, yhat_HMV_DK1 = walk_forward_validation(data_HMV_DK1, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_HMV_DK1)
# plot expected vs preducted
pyplot.plot(y_HMV_DK1, label='Expected')
pyplot.plot(yhat_HMV_DK1, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on LMV_DK2 data
values_LMV_DK2 = LMV[['DK2']].values
# transform the time series data into supervised learning
data_LMV_DK2 = series_to_supervised(values_LMV_DK2, n_in=24)
# evaluate
import time
start_time = time.time()
mae_LMV_DK2, y_LMV_DK2, yhat_LMV_DK2 = walk_forward_validation(data_LMV_DK2, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_LMV_DK2)
# plot expected vs preducted
pyplot.plot(y_LMV_DK2, label='Expected')
pyplot.plot(yhat_LMV_DK2, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on HMV_DK2
values_HMV_DK2 = HMV[['DK2']].values
# transform the time series data into supervised learning
data_HMV_DK2 = series_to_supervised(values_HMV_DK2, n_in=24)
# evaluate
import time
start_time = time.time()
mae_HMV_DK2, y_HMV_DK2, yhat_HMV_DK2 = walk_forward_validation(data_HMV_DK2, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_HMV_DK2)
# plot expected vs preducted
pyplot.plot(y_HMV_DK2, label='Expected')
pyplot.plot(yhat_HMV_DK2, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on LMV_NO2 data
values_LMV_NO2 = LMV[['NO2']].values
# transform the time series data into supervised learning
data_LMV_NO2 = series_to_supervised(values_LMV_NO2, n_in=24)
# evaluate
import time
start_time = time.time()
mae_LMV_NO2, y_LMV_NO2, yhat_LMV_NO2 = walk_forward_validation(data_LMV_NO2, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_LMV_NO2)
# plot expected vs preducted
pyplot.plot(y_LMV_NO2, label='Expected')
pyplot.plot(yhat_LMV_NO2, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on HMV_NO2
values_HMV_NO2 = HMV[['NO2']].values
# transform the time series data into supervised learning
data_HMV_NO2 = series_to_supervised(values_HMV_NO2, n_in=24)
# evaluate
import time
start_time = time.time()
mae_HMV_NO2, y_HMV_NO2, yhat_HMV_NO2 = walk_forward_validation(data_HMV_NO2, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_HMV_NO2)
# plot expected vs preducted
pyplot.plot(y_HMV_NO2, label='Expected')
pyplot.plot(yhat_HMV_NO2, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on LMV_SE3 data
values_LMV_SE3 = LMV[['SE3']].values
# transform the time series data into supervised learning
data_LMV_SE3 = series_to_supervised(values_LMV_SE3, n_in=24)
# evaluate
import time
start_time = time.time()
mae_LMV_SE3, y_LMV_SE3, yhat_LMV_SE3 = walk_forward_validation(data_LMV_SE3, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_LMV_SE3)
# plot expected vs preducted
pyplot.plot(y_LMV_SE3, label='Expected')
pyplot.plot(yhat_LMV_SE3, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on HMV_SE3
values_HMV_SE3 = HMV[['SE3']].values
# transform the time series data into supervised learning
data_HMV_SE3 = series_to_supervised(values_HMV_SE3, n_in=24)
# evaluate
import time
start_time = time.time()
mae_HMV_SE3, y_HMV_SE3, yhat_HMV_SE3 = walk_forward_validation(data_HMV_SE3, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_HMV_SE3)
# plot expected vs preducted
pyplot.plot(y_HMV_SE3, label='Expected')
pyplot.plot(yhat_HMV_SE3, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on LMV_SE4 data
values_LMV_SE4 = LMV[['SE4']].values
# transform the time series data into supervised learning
data_LMV_SE4 = series_to_supervised(values_LMV_SE4, n_in=24)
# evaluate
import time
start_time = time.time()
mae_LMV_SE4, y_LMV_SE4, yhat_LMV_SE4 = walk_forward_validation(data_LMV_SE4, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_LMV_SE4)
# plot expected vs preducted
pyplot.plot(y_LMV_SE4, label='Expected')
pyplot.plot(yhat_LMV_SE4, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Run XGBoost on HMV_SE4
values_HMV_SE4 = HMV[['SE4']].values
# transform the time series data into supervised learning
data_HMV_SE4 = series_to_supervised(values_HMV_SE4, n_in=24)
# evaluate
import time
start_time = time.time()
mae_HMV_SE4, y_HMV_SE4, yhat_HMV_SE4 = walk_forward_validation(data_HMV_SE4, 24)
print("--- %s seconds ---" % (time.time() - start_time))
print('MAE: %.3f' % mae_HMV_SE4)
# plot expected vs preducted
pyplot.plot(y_HMV_SE4, label='Expected')
pyplot.plot(yhat_HMV_SE4, label='Predicted')
pyplot.legend()
pyplot.show()




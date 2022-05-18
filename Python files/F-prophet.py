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

#%% Create Class for colour coding of ADF test_DE results to enhance readability in the terminal
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
    
#%% import data for plotting ALL forecast performance
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
HMV.reset_index(inplace=True)
HMV.rename(columns={'HourDK':'ds'},inplace=True)

DE = HMV[['ds', 'DE']]
DK1 = HMV[['ds', 'DK1']]
DK2 = HMV[['ds', 'DK2']]
NO2 = HMV[['ds', 'NO2']]
SE3 = HMV[['ds', 'SE3']]
SE4 = HMV[['ds', 'SE4']]

#%% rename to fit prophet format
DE.rename(columns={'DE':'y'}, inplace=True)
DK1.rename(columns={'DK1':'y'}, inplace=True)
DK2.rename(columns={'DK2':'y'}, inplace=True)
NO2.rename(columns={'NO2':'y'}, inplace=True)
SE3.rename(columns={'SE3':'y'}, inplace=True)
SE4.rename(columns={'SE4':'y'}, inplace=True)


#%%  import prophet
# check prophet version
from prophet import Prophet
# print version number
# print('Prophet %s' % prophet.__version__)

#%% define forecasting horizon
horizon = 24

#%% train_DE test_DE split
# we will go with a train_DE-test_DE split such that our test_DE set represents 168 Hours worth of data
train_DE =  DE[:len(DE)-horizon]
test_DE = DE[len(DE)-horizon:]
len(DE) == len(train_DE) + len(test_DE) # True

# forecast start and end
# obtain predicted results
start = len(train_DE)
end = len(train_DE)+len(test_DE)-1

#%% try facebook prophet model
import time
start_time = time.time()
# define the model
model_DE = Prophet()
# fit the model
model_DE.fit(train_DE)
time_DE_Prophet = (time.time() - start_time)

#%% create forecast
DE_fcst = model_DE.predict(df=test_DE)

#%%  create eval dataframe and append prophet predictions
DE_eval = test_DE
DE_eval.reset_index(inplace=True)
DE_eval.drop('index', axis = 1, inplace=True)
DE_eval['yhat']= DE_fcst['yhat']

#%% plot predictions
# define plot parameters
title='Prophet forecasting performance DE'
ylabel='Electricity Price'
xlabel=''

ax = DE_eval['y'].plot(legend=True,figsize=(20,6),title=title)
DE_eval['yhat'].plot(linestyle = '--', legend=True, color='black')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

#%% error metrics
from sklearn.metrics import mean_absolute_percentage_error as mape
error_DE_prophet = mape(DE_eval['y'], DE_eval['yhat'])*100

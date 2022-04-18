#%% import libraries
import pandas as pd
import seaborn as sns
import numpy as np

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
LMV = pd.read_csv('NP-LMV.csv', index_col='HourDK', parse_dates=True)
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 

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
    
#%% create series_to_supervised function to enable supervised training for XGBoost
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

#%% change to array type for supervised learning
for i in LMV.columns[0:]:
    text=f"LMV_new{i}=series_to_supervised(values, n_in=7)"
    exec(text)
    
#%% change to array type for supervised learning
for i in LMV.columns[0:]:
    text=f"LMV_new{i}=series_to_supervised(values, n_in=7)"
    exec(text)

#%% use series_to_supervised 
data =  series_to_supervised(LMV[['DE']].values, n_in=7)
data =  series_to_supervised(data, n_in=7)
LMV_new=series_to_supervised(values, n_in=7)





#%% import libraries
import pandas as pd
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

#%% Import dataset
LMV = pd.read_csv('NP-LMV.csv', index_col='HourDK', parse_dates=True)
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 

#%% import libraries for AR-based models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tools.eval_measures import mse,rmse     # for ETS Plots
from pmdarima import auto_arima 

#%% reduce LMV and HMV series load to enable auto arima
LMV_autoarima = LMV['2016-12-01':'2017-03-01']
HMV_autoarima = HMV['2021-12-01':'2022-03-01']

#%% Run autorima on all series and retrieve best model specification for forecasting
for series in LMV.columns[0:]:
    print(bcolors.UNDERLINE + 'Estimating Best ARIMA model for: ' + series + bcolors.ENDC)
    print(auto_arima(LMV[series]).summary())
    


    
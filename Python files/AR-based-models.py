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

#%% Import dataset
LMV = pd.read_csv('NP-LMV.csv', index_col='HourDK', parse_dates=True)
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 

#%% import libraries for AR-based models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tools.eval_measures import mse,rmse     # for ETS Plots
from pmdarima import auto_arima 

#%% reduce aseries load to enable auto arima
existing = ["DE", "DK1", "DK2", "SE3", "SE4", "NO2"]
new = ["DE_HMV", "DK1_HMV", "DK2_HMV", "SE3_HMV", "SE4_HMV", "NO2_HMV"]
for column in HMV.columns[0:]:
    column = HMV[column]
    
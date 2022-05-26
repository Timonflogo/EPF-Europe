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

#%% print distribution plots HMV
# fig, axes = plt.subplots(2)
fig, axes = plt.subplots(2, 3, figsize=(20,12))


sns.distplot(HMV['DE'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[0,0]
                 )
sns.distplot(HMV['DK1'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[0,1]
                 )
sns.distplot(HMV['DK2'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[0,2]
                 )
sns.distplot(HMV['SE3'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[1,0]
                 )
sns.distplot(HMV['SE4'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[1,1]
                 )
sns.distplot(HMV['NO2'], hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[1,2]
                 )

#%% print distribution plots LMV
# fig, axes = plt.subplots(2)
fig, axes = plt.subplots(2, 3, figsize=(20, 8))

fig.suptitle('NP-LMV Series distribution')


sns.distplot(LMV['DE'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[0,0]
                 )
sns.distplot(LMV['DK1'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[0,1]
                 )
sns.distplot(LMV['DK2'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[0,2]
                 )
sns.distplot(LMV['SE3'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[1,0]
                 )
sns.distplot(LMV['SE4'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[1,1]
                 )
sns.distplot(LMV['NO2'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 ax = axes[1,2]
                 )



#%% get descriptive statistics for NP_LMV
print("LMV descriptive statistics")
pd.options.display.max_columns = LMV.shape[1]
print(round(LMV.describe(include='all'),2))

#%% reduce NP_HMV series
HMV = HMV['2021-03-01':'2022-03-01']

#%% get descriptive statistics for NP_HMV
print("HMV descriptive statistics")
pd.options.display.max_columns = HMV.shape[1]
print(round(HMV.describe(include='all'),2))

#%% Create Class for colouyr coding of ADF test results to enhance readability in the terminal
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

#%% Define ADF test function
from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis{bcolors.ENDC}")
        print("Reject the null hypothesis")
        print(bcolors.OKBLUE + series.name + " Series Data has no unit root and is stationary" + bcolors.ENDC)
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print(bcolors.WARNING + series.name + " Series Data has a unit root and is non-stationary " + bcolors.ENDC)

#%% run ADF test on all HMV series     
print(bcolors.BOLD + bcolors.UNDERLINE + "RUN ADF TESTS ON HMV SERIES:" + bcolors.ENDC)  
for column in HMV.columns[0:]:
    print(bcolors.UNDERLINE + "ADF Testing for " + column + " Series of Type HMV:" + bcolors.ENDC)
    adf_test(HMV[column])
    
#%% run ADF test on all LMV series 
print(bcolors.BOLD + bcolors.UNDERLINE + "RUN ADF TESTS ON LMV SERIES:" + bcolors.ENDC) 
for column in LMV.columns[0:]:
    print(bcolors.UNDERLINE + "ADF Testing for " + column + " Series of Type LMV:" + bcolors.ENDC)
    adf_test(LMV[column])
    
#%% 
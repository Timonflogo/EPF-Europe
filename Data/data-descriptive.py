# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:37:05 2022

@author: timon
"""

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


#%% get descriptive statistics for NP_LMV
print("LMV descriptive statistics")
pd.options.display.max_columns = LMV.shape[1]
print(round(LMV.describe(include='all'),2))

#%% get descriptive statistics for NP_HMV
print("HMV descriptive statistics")
pd.options.display.max_columns = HMV.shape[1]
print(round(HMV.describe(include='all'),2))

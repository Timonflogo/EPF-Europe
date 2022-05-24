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

mpl.rcParams['figure.figsize'] = (16, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='hls', font_scale=1.4)

# set plotting parameters
rcParams['figure.figsize'] = 15, 20
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=2, linestyle='-')
# rcParams['text.usete'] = True

#%% import data
computing_time = pd.read_csv('C:/Users/timon/Documents/GitHub/EPF-Europe/Data/Time-compute-overall.csv').iloc[:,1:]

#%% plotting 
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

mpl.rcParams['figure.figsize'] = (16, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='hls', font_scale=1.4)

# set plotting parameters
rcParams['figure.figsize'] = 15, 20
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=2, linestyle='-')
# rcParams['text.usete'] = True

#%% import data
computing_time = pd.read_csv('C:/Users/timon/Documents/GitHub/EPF-Europe/Data/Time-compute-overall.csv').iloc[:,1:]

#%%
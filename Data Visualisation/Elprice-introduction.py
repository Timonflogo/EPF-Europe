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
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=1.5, font= 'Helvetica')

# set plotting parameters
rcParams['figure.figsize'] = 16, 8
rcParams['font.family'] = "serif"
# rcParams['text.usetex'] = True

#%% Import dataset
df = pd.read_csv('Elprices.csv', index_col='HourDK', parse_dates=True) 

#%% plot series 
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
df[['DK1', 'NO2', 'SE3']].plot(linewidth = 0.3, xlabel = 'Year', title="European bidding areas", subplots=True)

# resample by Month
# df['DK1'].resample('D').sum().plot(linewidth = 0.5, xlabel = 'Year')

#%% import dataset for renewable energy consumption
df1 = pd.read_csv('renewable-energy-consumption.csv').iloc[:-3]
# create barplot data 
df1 = df1[['Country', '2020']]
df1 = df1[df1.Country.isin(['Germany', 'Denmark', 'Sweden', 'Norway'])]
# plot barplot
fig = sns.barplot(x = "Country", y="2020", data=df1)
fig.set(ylabel="Share of renewable energy in total consumption")
plt.show()

# transpose
df2 = df1.T
# new header
new_header = df2.iloc[0] #grab the first row for the header
df2 = df2[1:] #take the data less the header row
df2.columns = new_header
# plot 
df2[['Germany','Norway', 'Denmark', 'Sweden']].plot()

#%% import libraries
import pandas as pd
import seaborn as sns

#%% Setup plotting environment
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

mpl.rcParams['figure.figsize'] = (20, 10)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=2)

# set plotting parameters
rcParams['figure.figsize'] = 15, 20 
rcParams['font.family'] = "sans-serif"
# rcParams['text.usetex'] = True

#%% Import dataset
df = pd.read_csv('Elprices.csv', index_col='HourDK', parse_dates=True) 

#%% plot only few series 
df['DK1'].plot(linewidth = 1, 
                legend = True,     
                xlabel = 'Year', 
                subplots=True, color = 'tab:blue')

# resample by Month
# df['DK1'].resample('D').sum().plot(linewidth = 0.5, xlabel = 'Year')

#%% plot all series 
fig, axes= plt.subplots(6,1)


df[['DK1']].plot(linewidth = 1, xlabel = 'Year', color = 'tab:blue',ax=axes[0],sharex = True)
df[['DK2']].plot(linewidth = 1, xlabel = 'Year', color = 'tab:blue',ax=axes[1],sharex = True)
df[['SE3']].plot(linewidth = 1, xlabel = 'Year', color = 'tab:blue',ax=axes[2])
df[['SE4']].plot(linewidth = 1, xlabel = 'Year', color = 'tab:blue',ax=axes[3])
df[['NO2']].plot(linewidth = 1, xlabel = 'Year', color = 'tab:blue',ax=axes[4])
df[['DE']].plot(linewidth = 1, xlabel = 'Year', color = 'tab:blue',ax=axes[5])
for i in range(len(axes)):
    axes[i].axvline(pd.Timestamp("2017-03-01"), color='red')
plt.show()

# resample by Month
# df['DK1'].resample('D').sum().plot(linewidth = 0.5, xlabel = 'Year')

#%% import dataset for renewable energy consumption
df1 = pd.read_csv('renewable-energy-consumption.csv').iloc[:-3]
# create barplot data 
df1 = df1[['Country', '2020']]
df1 = df1[df1.Country.isin(['Germany', 'Denmark', 'Sweden', 'Norway'])]
# plot barplot
fig = sns.barplot(x = "Country", y="2020", data=df1, color= "g")
fig.set(ylabel="Share of renewable energy in total consumption")
plt.show()

# # transpose
# df2 = df1.T
# # new header
# new_header = df2.iloc[0] #grab the first row for the header
# df2 = df2[1:] #take the data less the header row
# df2.columns = new_header
# # plot 
# df2[['Germany','Norway', 'Denmark', 'Sweden']].plot()

#%% create barplot with Matplotlib
df1 = pd.read_csv('renewable-energy-consumption.csv').iloc[:-3]
# create barplot data 
df1 = df1[['Country', '2020']]
df1 = df1[df1.Country.isin(['Germany', 'Denmark', 'Sweden', 'Norway'])]

# create y axis formatter to get percetnage on y axis
from matplotlib.ticker import FuncFormatter

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df1['Country'],df1['2020'])
formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
ax.yaxis.set_major_formatter(formatter)
plt.show()


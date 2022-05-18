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

#%% melt dataframe for boxplot
df = pd.melt(computing_time)

#%% create boxplot
ax = sns.boxplot(x="variable", y="value", data=df)
ax.set_xlabel("Model", fontsize = 20)
ax.set_ylabel("Computation in Seconds", fontsize = 20)

#%% import data for plotting of Statistical forecast performance
DE = pd.read_csv('test_eval_DE.csv').iloc[:,1:5]
DK1 = pd.read_csv('test_eval_DK1.csv').iloc[:,1:5]
DK2 = pd.read_csv('test_eval_DK2.csv').iloc[:,1:5]
NO2 = pd.read_csv('test_eval_NO2.csv').iloc[:,1:5]
SE3 = pd.read_csv('test_eval_SE3.csv').iloc[:,1:5]
SE4 = pd.read_csv('test_eval_SE4.csv').iloc[:,1:5]

#%% plot forecast performance for statistical models
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

ax1.plot(DE.iloc[:,0], marker = 'x', color='b')
ax1.plot(DE.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax1.plot(DE.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax1.plot(DE.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax1.set_ylabel('DE')
ax1.legend(DE, loc='upper center', bbox_to_anchor=(0.5, 1.45),
          ncol=4, fancybox=True)

ax2.plot(DK1.iloc[:,0], marker = 'x', color='b')
ax2.plot(DK1.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax2.plot(DK1.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax2.plot(DK1.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax2.set_ylabel('DK1')

ax3.plot(DK2.iloc[:,0], marker = 'x', color='b')
ax3.plot(DK2.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax3.plot(DK2.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax3.plot(DK2.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax3.set_ylabel('DK2')

ax4.plot(NO2.iloc[:,0], marker = 'x', color='b')
ax4.plot(NO2.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax4.plot(NO2.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax4.plot(NO2.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax4.set_ylabel('NO2')

ax5.plot(SE3.iloc[:,0], marker = 'x', color='b')
ax5.plot(SE3.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax5.plot(SE3.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax5.plot(SE3.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax5.set_ylabel('SE3')

ax6.plot(SE4.iloc[:,0], marker = 'x', color='b')
ax6.plot(SE4.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax6.plot(SE4.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax6.plot(SE4.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax6.set_ylabel('SE4')
ax6.set_xlabel('time (H)')

ax1.set_ylim([0, 400])
ax2.set_ylim([0, 400])
ax3.set_ylim([0, 400])
ax4.set_ylim([0, 400])
ax5.set_ylim([0, 400])
ax6.set_ylim([0, 400])


plt.show()

#%% import data for plotting of ML forecast performance
DE = pd.read_csv('test_eval_DE.csv').iloc[:,[1,5,6]]
DK1 = pd.read_csv('test_eval_DK1.csv').iloc[:,[1,5,6]]
DK2 = pd.read_csv('test_eval_DK2.csv').iloc[:,[1,5,6]]
NO2 = pd.read_csv('test_eval_NO2.csv').iloc[:,[1,5,6]]
SE3 = pd.read_csv('test_eval_SE3.csv').iloc[:,[1,5,6]]
SE4 = pd.read_csv('test_eval_SE4.csv').iloc[:,[1,5,6]]

#%% plot forecast performance for ML models
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

ax1.plot(DE.iloc[:,0], marker = 'x', color='b')
ax1.plot(DE.iloc[:,1], marker = 'x',color='c', linestyle = '--')
ax1.plot(DE.iloc[:,2], marker = 'x',color='r', linestyle = '--')
ax1.set_ylabel('DE')
ax1.legend(DE, loc='upper center', bbox_to_anchor=(0.5, 1.45),
          ncol=4, fancybox=True)

ax2.plot(DK1.iloc[:,0], marker = 'x', color='b')
ax2.plot(DK1.iloc[:,1], marker = 'x',color='c', linestyle = '--')
ax2.plot(DK1.iloc[:,2], marker = 'x',color='r', linestyle = '--')
ax2.set_ylabel('DK1')

ax3.plot(DK2.iloc[:,0], marker = 'x', color='b')
ax3.plot(DK2.iloc[:,1], marker = 'x',color='c', linestyle = '--')
ax3.plot(DK2.iloc[:,2], marker = 'x',color='r', linestyle = '--')
ax3.set_ylabel('DK2')

ax4.plot(NO2.iloc[:,0], marker = 'x', color='b')
ax4.plot(NO2.iloc[:,1], marker = 'x',color='c', linestyle = '--')
ax4.plot(NO2.iloc[:,2], marker = 'x',color='r', linestyle = '--')
ax4.set_ylabel('NO2')

ax5.plot(SE3.iloc[:,0], marker = 'x', color='b')
ax5.plot(SE3.iloc[:,1], marker = 'x',color='c', linestyle = '--')
ax5.plot(SE3.iloc[:,2], marker = 'x',color='r', linestyle = '--')
ax5.set_ylabel('SE3')

ax6.plot(SE4.iloc[:,0], marker = 'x', color='b')
ax6.plot(SE4.iloc[:,1], marker = 'x',color='c', linestyle = '--')
ax6.plot(SE4.iloc[:,2], marker = 'x',color='r', linestyle = '--')
ax6.set_ylabel('SE4')
ax6.set_xlabel('time (H)')

ax1.set_ylim([0, 400])
ax2.set_ylim([0, 400])
ax3.set_ylim([0, 400])
ax4.set_ylim([0, 400])
ax5.set_ylim([0, 400])
ax6.set_ylim([0, 400])


plt.show()


#%% import data for plotting of DL forecast performance
DE = pd.read_csv('test_eval_DE.csv').iloc[:,[1,7,8]]
DK1 = pd.read_csv('test_eval_DK1.csv').iloc[:,[1,7,8]]
DK2 = pd.read_csv('test_eval_DK2.csv').iloc[:,[1,7,8]]
NO2 = pd.read_csv('test_eval_NO2.csv').iloc[:,[1,7,8]]
SE3 = pd.read_csv('test_eval_SE3.csv').iloc[:,[1,7,8]]
SE4 = pd.read_csv('test_eval_SE4.csv').iloc[:,[1,7,8]]

#%% plot forecast performance for DL models
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

ax1.plot(DE.iloc[:,0], marker = 'x', color='b')
ax1.plot(DE.iloc[:,1], marker = 'x',color='k', linestyle = '--')
ax1.plot(DE.iloc[:,2], marker = 'x',color='peru', linestyle = '--')
ax1.set_ylabel('DE')
ax1.legend(DE, loc='upper center', bbox_to_anchor=(0.5, 1.45),
          ncol=4, fancybox=True)

ax2.plot(DK1.iloc[:,0], marker = 'x', color='b')
ax2.plot(DK1.iloc[:,1], marker = 'x',color='k', linestyle = '--')
ax2.plot(DK1.iloc[:,2], marker = 'x',color='peru', linestyle = '--')
ax2.set_ylabel('DK1')

ax3.plot(DK2.iloc[:,0], marker = 'x', color='b')
ax3.plot(DK2.iloc[:,1], marker = 'x',color='k', linestyle = '--')
ax3.plot(DK2.iloc[:,2], marker = 'x',color='peru', linestyle = '--')
ax3.set_ylabel('DK2')

ax4.plot(NO2.iloc[:,0], marker = 'x', color='b')
ax4.plot(NO2.iloc[:,1], marker = 'x',color='k', linestyle = '--')
ax4.plot(NO2.iloc[:,2], marker = 'x',color='peru', linestyle = '--')
ax4.set_ylabel('NO2')

ax5.plot(SE3.iloc[:,0], marker = 'x', color='b')
ax5.plot(SE3.iloc[:,1], marker = 'x',color='k', linestyle = '--')
ax5.plot(SE3.iloc[:,2], marker = 'x',color='peru', linestyle = '--')
ax5.set_ylabel('SE3')

ax6.plot(SE4.iloc[:,0], marker = 'x', color='b')
ax6.plot(SE4.iloc[:,1], marker = 'x',color='k', linestyle = '--')
ax6.plot(SE4.iloc[:,2], marker = 'x',color='peru', linestyle = '--')
ax6.set_ylabel('SE4')
ax6.set_xlabel('time (H)')


ax1.set_ylim([0, 400])
ax2.set_ylim([0, 400])
ax3.set_ylim([0, 400])
ax4.set_ylim([0, 400])
ax5.set_ylim([0, 400])
ax6.set_ylim([0, 400])

plt.show()


#%% import data for plotting ALL forecast performance
DE = pd.read_csv('test_eval_DE.csv').iloc[:,1:]
DK1 = pd.read_csv('test_eval_DK1.csv').iloc[:,1:]
DK2 = pd.read_csv('test_eval_DK2.csv').iloc[:,1:]
NO2 = pd.read_csv('test_eval_NO2.csv').iloc[:,1:]
SE3 = pd.read_csv('test_eval_SE3.csv').iloc[:,1:]
SE4 = pd.read_csv('test_eval_SE4.csv').iloc[:,1:]

#%%
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

ax1.plot(DE.iloc[:,0], marker = 'x', color='b')
ax1.plot(DE.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax1.plot(DE.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax1.plot(DE.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax1.plot(DE.iloc[:,4], marker = 'x',color='c', linestyle = '--')
ax1.plot(DE.iloc[:,5], marker = 'x',color='r', linestyle = '--')
ax1.plot(DE.iloc[:,6], marker = 'x',color='k', linestyle = '--')
ax1.plot(DE.iloc[:,7], marker = 'x',color='peru', linestyle = '--')
ax1.set_ylabel('DE')
ax1.legend(DE, loc='upper center', bbox_to_anchor=(0.5, 1.45),
          ncol=4, fancybox=True)
ax1.set_ylim([0, 400])

ax2.plot(DK1.iloc[:,0], marker = 'x', color='b')
ax2.plot(DK1.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax2.plot(DK1.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax2.plot(DK1.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax2.plot(DK1.iloc[:,4], marker = 'x',color='c', linestyle = '--')
ax2.plot(DK1.iloc[:,5], marker = 'x',color='r', linestyle = '--')
ax2.plot(DK1.iloc[:,6], marker = 'x',color='k', linestyle = '--')
ax2.plot(DK1.iloc[:,7], marker = 'x',color='peru', linestyle = '--')
ax2.set_ylabel('DK1')
ax2.set_ylim([0, 400])

ax3.plot(DK2.iloc[:,0], marker = 'x', color='b')
ax3.plot(DK2.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax3.plot(DK2.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax3.plot(DK2.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax3.plot(DK2.iloc[:,4], marker = 'x',color='c', linestyle = '--')
ax3.plot(DK2.iloc[:,5], marker = 'x',color='r', linestyle = '--')
ax3.plot(DK2.iloc[:,6], marker = 'x',color='k', linestyle = '--')
ax3.plot(DK2.iloc[:,7], marker = 'x',color='peru', linestyle = '--')
ax3.set_ylabel('DK2')
ax3.set_ylim([0, 400])

ax4.plot(NO2.iloc[:,0], marker = 'x', color='b')
ax4.plot(NO2.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax4.plot(NO2.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax4.plot(NO2.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax4.plot(NO2.iloc[:,4], marker = 'x',color='c', linestyle = '--')
ax4.plot(NO2.iloc[:,5], marker = 'x',color='r', linestyle = '--')
ax4.plot(NO2.iloc[:,6], marker = 'x',color='k', linestyle = '--')
ax4.plot(NO2.iloc[:,7], marker = 'x',color='peru', linestyle = '--')
ax4.set_ylim([0, 400])

ax5.plot(SE3.iloc[:,0], marker = 'x', color='b')
ax5.plot(SE3.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax5.plot(SE3.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax5.plot(SE3.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax5.plot(SE3.iloc[:,4], marker = 'x',color='c', linestyle = '--')
ax5.plot(SE3.iloc[:,5], marker = 'x',color='r', linestyle = '--')
ax5.plot(SE3.iloc[:,6], marker = 'x',color='k', linestyle = '--')
ax5.plot(SE3.iloc[:,7], marker = 'x',color='peru', linestyle = '--')
ax5.set_ylabel('SE3')
ax5.set_ylim([0, 400])

ax6.plot(SE4.iloc[:,0], marker = 'x', color='b')
ax6.plot(SE4.iloc[:,1], marker = 'x',color='m', linestyle = '--')
ax6.plot(SE4.iloc[:,2], marker = 'x',color='g', linestyle = '--')
ax6.plot(SE4.iloc[:,3], marker = 'x',color='y', linestyle = '--')
ax6.plot(SE4.iloc[:,4], marker = 'x',color='c', linestyle = '--')
ax6.plot(SE4.iloc[:,5], marker = 'x',color='r', linestyle = '--')
ax6.plot(SE4.iloc[:,6], marker = 'x',color='k', linestyle = '--')
ax6.plot(SE4.iloc[:,7], marker = 'x',color='peru', linestyle = '--')
ax6.set_ylabel('SE4')
ax6.set_xlabel('time (H)')
ax6.set_ylim([0, 400])

plt.show()

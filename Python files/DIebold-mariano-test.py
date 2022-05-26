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
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='hls', font_scale=1.4)

# set plotting parameters
rcParams['figure.figsize'] = 8, 6
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=2, linestyle='-')
# rcParams['text.usete'] = True

#%% import data
DE = pd.read_csv('test_eval_DE.csv')
DK1 = pd.read_csv('test_eval_DK1.csv')
DK2 = pd.read_csv('test_eval_DK2.csv')
NO2 = pd.read_csv('test_eval_NO2.csv')
SE3 = pd.read_csv('test_eval_SE3.csv')
SE4 = pd.read_csv('test_eval_SE4.csv')

#%% import Diebold mariano
from epftoolbox.evaluation import DM, plot_multivariate_DM_test

#%% Modify Heatmap
def plot_multivariate_DM_test(real_price, forecasts, norm=1, title='DM test', savefig=False, path=''):
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1, 24), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24), 
                                                  norm=norm, version='multivariate')
    
    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    
    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(title + '.png', dpi=300)
        plt.savefig(title + '.eps')
    
    plt.show()

#%% DE Diebold-Mariano
real_price_DE = DE.loc[:, ['DE']]
forecasts_DE = DE.loc[:, ['AR', 'ARIMA', 'SARIMA', 'XGBoost', 'SVR', 'LSTM', 'GRU']]

plot_multivariate_DM_test(real_price=real_price_DE, forecasts=forecasts_DE, title='DM Test DE')

#%% DK1 Diebold-Mariano
real_price_DK1 = DK1.loc[:, ['DK1']]
forecasts_DK1 = DK1.loc[:, ['AR', 'ARIMA', 'SARIMA', 'XGBoost', 'SVR', 'LSTM', 'GRU']]

plot_multivariate_DM_test(real_price=real_price_DK1, forecasts=forecasts_DK1, title='DM Test DK1')

#%% DK2 Diebold-Mariano
real_price_DK2 = DK2.loc[:, ['DK2']]
forecasts_DK2 = DK2.loc[:, ['AR', 'ARIMA', 'SARIMA', 'XGBoost', 'SVR', 'LSTM', 'GRU']]

plot_multivariate_DM_test(real_price=real_price_DK2, forecasts=forecasts_DK2, title='DM Test DK2')

#%% NO2 Diebold-Mariano
real_price_NO2 = NO2.loc[:, ['NO2']]
forecasts_NO2 = NO2.loc[:, ['AR', 'ARIMA', 'SARIMA', 'XGBoost', 'SVR', 'LSTM', 'GRU']]

plot_multivariate_DM_test(real_price=real_price_NO2, forecasts=forecasts_NO2, title='DM Test NO2')

#%% SE3 Diebold-Mariano
real_price_SE3 = SE3.loc[:, ['SE3']]
forecasts_SE3 = SE3.loc[:, ['AR', 'ARIMA', 'SARIMA', 'XGBoost', 'SVR', 'LSTM', 'GRU']]

plot_multivariate_DM_test(real_price=real_price_SE3, forecasts=forecasts_SE3, title='DM Test SE3')

#%% SE4 Diebold-Mariano
real_price_SE4 = SE4.loc[:, ['SE4']]
forecasts_SE4 = SE4.loc[:, ['AR', 'ARIMA', 'SARIMA', 'XGBoost', 'SVR', 'LSTM', 'GRU']]

plot_multivariate_DM_test(real_price=real_price_SE4, forecasts=forecasts_SE4, title='DM Test SE4')

#%% subplot all Diebold Mariano tests
fig, axs = plt.subplots(2, 3)
axs[0, 0].plot_multivariate_DM_test(real_price=real_price_DE, forecasts=forecasts_DE, title='DM Test DE')
axs[1, 0].plot_multivariate_DM_test(real_price=real_price_DK1, forecasts=forecasts_DK1, title='DM Test DK1')
axs[0, 1].plot_multivariate_DM_test(real_price=real_price_DK2, forecasts=forecasts_DK2, title='DM Test DK2')
axs[1, 1].plot_multivariate_DM_test(real_price=real_price_NO2, forecasts=forecasts_NO2, title='DM Test NO2')
axs[0, 2].plot_multivariate_DM_test(real_price=real_price_SE3, forecasts=forecasts_SE3, title='DM Test SE3')
axs[1, 2].plot_multivariate_DM_test(real_price=real_price_SE4, forecasts=forecasts_SE4, title='DM Test SE4')
fig.tight_layout()

#%% subplot all Diebold Mariano tests
fig, axs = plt.subplots(1, 2)
ax1 = plot_multivariate_DM_test(real_price=real_price_DE, forecasts=forecasts_DE, title='DM Test DE')
ax2 = plot_multivariate_DM_test(real_price=real_price_DK1, forecasts=forecasts_DK1, title='DM Test DK1')

fig.tight_layout()
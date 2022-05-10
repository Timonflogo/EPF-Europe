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
rcParams['figure.figsize'] = 20, 5  
rcParams['font.family'] = "sans-serif"
# rcParams['text.usetex'] = True

#%% sigmoid function
xsig = np.linspace(-10, 10, 1000)
ysig = 1 / (1 + np.exp(-xsig) )

plt.figure(figsize=(10, 5))
plt.plot(xsig, ysig)
plt.legend(['sigmoid function'])
plt.show()

#%% Tanh function
xtanh = np.linspace(-10, 10, 1000)
ytanh = ( 2 / (1 + np.exp(-2*xtanh) ) ) -1

plt.figure(figsize=(10, 5))
plt.plot(xtanh, ytanh)
plt.legend(['Tanh (hyperbolic tangent)'])
plt.show()

#%% ReLU function
xrel = np.linspace(-10, 10, 1000)
yrel = np.maximum(0, xrel)

plt.figure(figsize=(10, 5))
plt.plot(xrel, yrel)
plt.legend(['Relu'])
plt.show()

#%% create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(xsig, ysig, 'tab:red', linewidth = 5)
ax1.set_title('Sigmoid')
ax2.plot(xtanh, ytanh, 'tab:blue', linewidth = 5)
ax2.set_title('Tanh')
ax3.plot(xrel, yrel, 'tab:green', linewidth = 5)
ax3.set_title('ReLU')
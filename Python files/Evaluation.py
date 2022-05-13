#%% plot predictions AR
# reset index for plotting
test_eval.reset_index(inplace=True)
test_eval.drop('index',axis=1, inplace=True)
# define plot parameters
title='Statistical forecasting performance DE'
ylabel='Electricity Price'
xlabel=''

ax = test_eval['DE'].plot(legend=True,figsize=(20,6),title=title)
test_eval['AR(4,0,0)'].plot(linestyle = '--', legend=True, color='orange')
test_eval['ARIMA(4,1,3)'].plot(linestyle = '--', legend=True, color='green')
test_eval['SARIMA(1,1,0)(2,0,0,24)'].plot(linestyle = '--', legend=True,color='purple')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

#%% plot Statistical and ml
title='ML forecasting performance DE'
ylabel='Electricity Price'
xlabel=''

ax = test_eval['DE'].plot(legend=True,figsize=(20,12),title=title)
test_eval['AR(4,0,0)'].plot(linestyle = '--', legend=True, color='orange')
test_eval['ARIMA(4,1,3)'].plot(linestyle = '--', legend=True, color='green')
test_eval['SARIMA(1,1,0)(2,0,0,24)'].plot(linestyle = '--', legend=True,color='purple')
test_eval['SVR'].plot(linestyle = '--', legend=True, color='magenta')
test_eval['XGBoost'].plot(linestyle = '--', legend=True, color='cyan')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

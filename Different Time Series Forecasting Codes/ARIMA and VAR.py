#!/usr/bin/env python
# coding: utf-8

# # ARIMA 

# ## Import the data and the librarires

# In[1]:


import os
cwd = os.getcwd()
print(cwd)

chdir=os.chdir("D:\Folder D/New folder/WERK Student/EXCEL/")
print(chdir)


# In[41]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import pmdarima as pm
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic


# In[42]:


df = pd.read_csv('201102_DataSet Forecast_0.01.csv',sep=';',
                 header=0, parse_dates = ['transaction_date'], index_col = ['transaction_date'])


# ## Explorarty Data Analysis

# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


# Choose the highest demanded Ware House
print(df.warehouse_id.value_counts())


# In[7]:


# Choose the Highest Demanded peoduct from the 12 warehouse
df12=df[df['warehouse_id']==12]
print(df12.product_id.value_counts())
df342=df12[df12['product_id']==342]
print(df342.shape)
print(df342.columns)
df342=df342.drop(['warehouse_id','product_id'],axis=1)


# In[8]:


print(df342.head())
print(df342.shape)


# ## Apply forecasting on the df342

# ### Check the Stationarity of the df342, and have an assumed(p.d.q)

# In[9]:


rolling_mean = df342.rolling(window = 12).mean()
rolling_std = df342.rolling(window = 12).std()
plt.plot(df342, color = 'blue', label = 'Original')
plt.plot(rolling_mean['quantity'], color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std['quantity'], color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()


# In[10]:


# Applying adfuller test, with the null hypothesis if the p/values is less than 0.05 it means that it is stationary
result = adfuller(df342)
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
    # the model is stationary , and we don't need to differentiate at all the d=0


# In[11]:


# Applying the previous two cells in a function
def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['quantity'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        
get_stationarity(df342)


# In[12]:


# Remove the time index, so we can plot
df342.reset_index(inplace=True)
print(df342.head())


# In[13]:


#The right order of differencing is the minimum differencing required to get a near-stationary series which 
#roams around a defined mean and the ACF plot reaches to zero fairly quick.
#If the autocorrelations are positive for many number of lags (10 or more), then the series needs further differencing. On the other hand, if the lag 1 autocorrelation itself is too negative,
#then the series is probably over-differenced.

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df342.quantity); axes[0, 0].set_title('Original Series')
plot_acf(df342.quantity, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df342.quantity.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df342.quantity.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df342.quantity.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df342.quantity.diff().diff().dropna(), ax=axes[2, 1])

plt.show()


# In[14]:


# Select the best P,we can get it by ploting the PACF , but also ACF is helpful
# PACF plot of 1 differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 3, sharex=True)
axes[0].plot(df342.quantity.diff()); axes[0].set_title('1 Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df342.quantity.diff().dropna(), ax=axes[1])
axes[2].set(ylim=(0,5))
plot_acf(df342.quantity.diff().dropna(), ax=axes[2])

plt.show()
# I believe that i can select p=1 because in the ACF , it seems it starts to tail off in the 1 lag


# In[15]:


# Select the best q, we can get it by ploting the ACF , but also PACF is helpful
fig, axes = plt.subplots(1, 3, sharex=True)
axes[0].plot(df342.quantity.diff()); axes[0].set_title('1 Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df342.quantity.diff().dropna(), ax=axes[1])
axes[2].set(ylim=(0,5))
plot_pacf(df342.quantity.diff().dropna(), ax=axes[2])

plt.show()
# I believe that i can select q=1 because in the PACF , it seems it starts to tail off in the 0 lag


# In[16]:


#If your series is slightly under differenced, adding one or more additional AR terms usually makes it up. Likewise,
#if it is slightly over-differenced, try adding an additional MA term.

# 1,0,1 ARIMA Model
model = ARIMA(df342.quantity, order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[17]:


#Let’s plot the residuals to ensure there are no patterns (that is, look for constant mean and variance).

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[18]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()
#When you set dynamic=False the in-sample lagged values are used for prediction


# ### Try other number of p.d.q

# In[19]:


# Now we will do the Cross -Validation Times series decomposition
# Create Training and Test
train = df342.quantity[:17]
test = df342.quantity[17:]


# In[20]:


# Build Model, to see if we applied the previous assumed p.d.q how the forecast will be  
model = ARIMA(train, order=(1, 1, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(6, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[21]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)


# In[22]:


#So, what I am going to do is to increase the order of differencing to two, that is set d=2 and iteratively increase p to up to 5 and then q up to 5 to see which model gives least AIC 
#and also look for a chart that gives closer actuals and forecasts
#While doing this, I keep an eye on the P values of the AR and MA terms in the model summary.
#They should be as close to zero, ideally, less than 0.05.

# I think i have to build a for loop to figure out which one is the best, along low AIC and BIC
# Build Model
model = ARIMA(train, order=(3, 2, 0))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(6, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[23]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)


# ### Apply Cross Validation to get the best (p.d.q) of df342

# In[30]:


#auto_arima() uses a stepwise approach to search multiple combinations of p,d,q parameters
#and chooses the best model that has the least AIC.

model = pm.auto_arima(df342.quantity, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[25]:


#Model Diagnostic: it is a way to check if the model selected is the best 
#we see the residuals of the model  by mean absolute error , that it should be uncorrelated white noise gaussian , noise centered on zero
# Fit model
model = sm.tsa.statespace.SARIMAX(df342.quantity, order=(3,1,0))
results = model.fit()
# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))
# Print mean absolute error
print(mae)
results.plot_diagnostics()
plt.show()


# In[26]:


# Build Model, to see if we applied the previous assumed p.d.q how the forecast will be  
model = ARIMA(train, order=(3, 1, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(6, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[27]:


print('the predicted ones',fc)
print('the real values are',test.values)


# In[28]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

# Around 13.5% MAPE implies the model is about 86.5% accurate in predicting the next 6 observations.


# In[31]:


# Forecast
n_periods = 6
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df342.quantity), len(df342.quantity)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df342.quantity)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()


# In[43]:


# Remove the time index, so we can plot
#df.reset_index(inplace=True)
print(df.head())


# ## Applying VAR ( multiVariate time series) on the Whole Data

# In[45]:


print(df.head())
print(df.shape)


# ### Plot each column

# In[46]:


# Plot
fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# ### See the Causality of each column with others

# In[47]:


from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(df, variables = df.columns) 


# ### Cointegration to see if there is relationship

# In[48]:


#If a given p-value is < significance level (0.05), then,
#the corresponding X series (column) causes the Y (row).
# for example row 1 column 2 = 0.2306 which means that product id causes warehouse id

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)

#When two or more time series are cointegrated,
#it means they have a long run, statistically significant relationship.


# ### Train , test split

# In[49]:


df_train = df[:137162]
df_test = df[137162:]
# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)


# In[58]:


print(df_train[-3:])


# ### Check the best p order

# In[50]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


# In[51]:


# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[52]:


model = VAR(df_train)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


# In[53]:


#An alternate method to choose the order(p) of the VAR models is to use the model.select_order(maxlags)method.

#The selected order(p) is the order that gives the lowest ‘AIC’, ‘BIC’, ‘FPE’ and ‘HQIC’ scores.

x = model.select_order(maxlags=12)
x.summary()


# ### Train the VAR Model

# In[54]:


model_fitted = model.fit(3)
model_fitted.summary()


# ### Check for Serial Correlation of Residuals

# In[56]:


#If there is any correlation left in the residuals, then, 
#there is some pattern in the time series that is still left to be explained by the model.
#In that case, the typical course of action is to either increase the order of the model or induce more predictors 
#into the system or look for a different algorithm to model the time series.

#The value of this statistic can vary between 0 and 4. The closer it is to the value 2,
#then there is no significant serial correlation. The closer to 0,
#there is a positive serial correlation, and the closer it is to 4 implies negative serial correlation

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print((col), ':', round(val, 2))


# ### Forecast

# In[70]:


# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order) 

# Input data for forecasting
forecast_input = df_train.values[-45721:]
forecast_input


# In[71]:


# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=45721)
df_forecast = pd.DataFrame(fc, index=df.index[-45721:], columns=df.columns + '_2d')
df_forecast


# ### Plot the Forecast and the Actual

# In[72]:


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


# In[73]:


df_results = invert_transformation(df_train, df_forecast, second_diff=True)        


# In[74]:


print(df_results.columns)


# In[75]:


fig, axes = plt.subplots(nrows=int(len(df.columns)), ncols=1, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-3:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# ### Evaluate the Forecast

# In[77]:


from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

print('Forecast Accuracy of: Quantity')
accuracy_prod = forecast_accuracy(df_results['quantity_forecast'].values, df_test['quantity'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))

print('\nForecast Accuracy of: product_id')
accuracy_prod = forecast_accuracy(df_results['product_id_forecast'].values, df_test['product_id'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))

print('\nForecast Accuracy of: Ware_house_id')
accuracy_prod = forecast_accuracy(df_results['warehouse_id_forecast'].values, df_test['warehouse_id'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


# ## Applying ARIMA for each column alone

# ### FOrecasting the Quantity Column

# In[88]:


#auto_arima() uses a stepwise approach to search multiple combinations of p,d,q parameters
#and chooses the best model that has the least AIC.

model = pm.auto_arima(df.quantity, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[ ]:


#First of all, set the index of the Dataset as Date_Time. Secondly,
#ARIMA works on a univariate series. So you can either deal with each time series individually 
#or use a different forecasting model For multivariate time series, you can use VAR (Vector autoregression) model
#VARMA


# In[79]:


#Model Diagnostic: it is a way to check if the model selected is the best 
#we see the residuals of the model  by mean absolute error , that it should be uncorrelated white noise gaussian , noise centered on zero
# Fit model
model = sm.tsa.statespace.SARIMAX(df.quantity, order=(5,0,0))
results = model.fit()
# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))
# Print mean absolute error
print(mae)
results.plot_diagnostics()
plt.show()


# In[83]:


# Create Training and Test
train = df.quantity[:137162]
test = df.quantity[137162:]
print(train.head())


# In[84]:


# Build Model, to see if we applied the previous assumed p.d.q how the forecast will be  
# Now we will do the Cross -Validation Times series decomposition

model = ARIMA(train, order=(5, 0, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(45721, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[85]:


print('the predicted ones',fc)
print('the real values are',test.values)


# In[86]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

# Around 24.4% MAPE implies the model is about 75.6% accurate in predicting the next 45721 observations.


# In[91]:


# Forecast
n_periods = 30
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df342.quantity), len(df342.quantity)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df342.quantity)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()


# ### Forecasting the Product_id Column

# In[100]:


#auto_arima() uses a stepwise approach to search multiple combinations of p,d,q parameters
#and chooses the best model that has the least AIC.

model = pm.auto_arima(df.product_id, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[ ]:


#First of all, set the index of the Dataset as Date_Time. Secondly,
#ARIMA works on a univariate series. So you can either deal with each time series individually 
#or use a different forecasting model For multivariate time series, you can use VAR (Vector autoregression) model
#VARMA


# In[93]:


#Model Diagnostic: it is a way to check if the model selected is the best 
#we see the residuals of the model  by mean absolute error , that it should be uncorrelated white noise gaussian , noise centered on zero
# Fit model
model = sm.tsa.statespace.SARIMAX(df.product_id, order=(1,0,0))
results = model.fit()
# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))
# Print mean absolute error
print(mae)
results.plot_diagnostics()
plt.show()


# In[95]:


# Create Training and Test
train = df.product_id[:137162]
test = df.product_id[137162:]
print(train.head())


# In[96]:


# Build Model, to see if we applied the previous assumed p.d.q how the forecast will be  
# Now we will do the Cross -Validation Times series decomposition

model = ARIMA(train, order=(1, 0, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(45721, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[97]:


print('the predicted ones',fc)
print('the real values are',test.values)


# In[98]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

# The result of the product id is really bad


# In[101]:


# Forecast
n_periods = 4572
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df342.quantity), len(df342.quantity)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df342.quantity)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()


# ### Forecasting the Ware_house_id 

# In[102]:


#auto_arima() uses a stepwise approach to search multiple combinations of p,d,q parameters
#and chooses the best model that has the least AIC.

model = pm.auto_arima(df.warehouse_id, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[ ]:


#First of all, set the index of the Dataset as Date_Time. Secondly,
#ARIMA works on a univariate series. So you can either deal with each time series individually 
#or use a different forecasting model For multivariate time series, you can use VAR (Vector autoregression) model
#VARMA


# In[103]:


#Model Diagnostic: it is a way to check if the model selected is the best 
#we see the residuals of the model  by mean absolute error , that it should be uncorrelated white noise gaussian , noise centered on zero
# Fit model
model = sm.tsa.statespace.SARIMAX(df.quantity, order=(4,0,1))
results = model.fit()
# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))
# Print mean absolute error
print(mae)
results.plot_diagnostics()
plt.show()


# In[104]:


# Create Training and Test
train = df.warehouse_id[:137162]
test = df.warehouse_id[137162:]
print(train.head())


# In[105]:


# Build Model, to see if we applied the previous assumed p.d.q how the forecast will be  
# Now we will do the Cross -Validation Times series decomposition

model = ARIMA(train, order=(5, 0, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(45721, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[106]:


print('the predicted ones',fc)
print('the real values are',test.values)


# In[107]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

# Around 24.4% MAPE implies the model is about 75.6% accurate in predicting the next 45721 observations.


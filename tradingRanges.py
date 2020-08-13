# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:29:20 2020

@author: freel
"""

# -*- coding: utf-8 -*-
"""
Trading Range Analysis
todo: predict returns and vol from trailing levels, predict drawdown via logistic regression, train keras model to predict outputs
-Create a volatilty-driven range that incorporates trailing volatility, volume, and prices. 
-FFT amplitude & phase over time
-Hurst Exponent
-Different speeds
-Run over different time scales
-add a breakpoint and step through fracdiff
-ML model with inputs of price and volume and outputs of trading range
-minimize fractional dimension while preserving stationarity at high lag and fixed data range
-seasonality by asset?
-logistic regression model
-vol driven trend signal?
-defensive/offensive outperformance switching a la ATAC fund
-implied vol discount
"""

#import external libraries
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import nolds
import pickle

import fracDiffTest as frac


def loadDimensions(filename):
    dimensionFile = open(filename, 'rb')
    tmp = pickle.load(dimensionFile)
    dimensionFile.close()
    return tmp

def scanAndStore(window, filename):
    dimensionFile = open(filename, 'wb')
    res = frac.fracDiffScan(tickerDF, 'logPx', test_stocks, window, .9, 50, .01, 200)
    pickle.dump(res, dimensionFile)
    dimensionFile.close()
    return res


# def fractionalWeights(k, d):
#     w = np.array([1.0], dtype=float)
#     for i in range(1,k):
#         w = np.append(w, -1*w[i-1]*((d-i+1)/i))
#     print(w.sum())
#     return w

# #sum asymptotically approaches 0. It is 1 + series of negative numbers
# def fractionalWeightsCutoff(d, cutoff):
#     w = np.array([1.0], dtype=float)
#     i=1
#     while (w.sum()>cutoff):
#         w = np.append(w, -1*w[i-1]*((d-i+1)/i))
#         i+=1
#         print(w.sum())
#     return w

# def fracDiff(input_data, test_col, fraction, cutoffWindow, afdMaxLag):
#     temp_data = pd.Series()
#     weights = fractionalWeights(cutoffWindow,fraction)
#     for i in range (cutoffWindow,input_data.shape[0]):
#         temp_data[input_data.index[i]] = (input_data.loc[:,test_col].iloc[-1*cutoffWindow+i:i]).dot(np.flipud(weights))
#         #print(-1*cutoffWindowDaily+i)
#     #print(i)
#     result = adfuller(temp_data, autolag='AIC')
#     return temp_data, result[1]


#Pick stocks for analysis
test_stocks = ['SPY', 'TLT', 'IEF','TOTL','BSV','XLU','XLP','DEF','QQQ','GLD','SPLV','MBB','QUAL','SPHD','LQD','VCLT']
tickerData = {}
tickerDF = {}
averageChange = {}
hurstMod = {}
recursiveHurst = {}

for ticker in test_stocks:
    tickerData[ticker] = yf.Ticker(ticker)
    #get the historical prices for this ticker
    tickerDF[ticker] = tickerData[ticker].history(period='1d', start='1980-1-1', end='2020-12-31')
 
#for each ticker, calculate the various measures used for analysis
for ticker in test_stocks: 
    tickerDF[ticker].loc[:,'sumDiv'] = tickerDF[ticker].loc[:,'Dividends'].cumsum()
    tickerDF[ticker].loc[:,'logPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
    tickerDF[ticker].loc[:,'logDiff'] = np.log(tickerDF[ticker].loc[:,'Close']).diff()
    tickerDF[ticker].loc[:,'vol30'] = tickerDF[ticker].loc[:,'logDiff'].rolling(30).std()
    tickerDF[ticker].loc[:,'ret30'] = tickerDF[ticker].loc[:,'logDiff'].rolling(30).sum()
    tickerDF[ticker].loc[:,'vol15'] = tickerDF[ticker].loc[:,'logDiff'].rolling(15).std()
    tickerDF[ticker].loc[:,'ret15'] = tickerDF[ticker].loc[:,'logDiff'].rolling(15).sum()
    tickerDF[ticker].loc[:,'vol5'] = tickerDF[ticker].loc[:,'logDiff'].rolling(5).std()
    tickerDF[ticker].loc[:,'ret5'] = tickerDF[ticker].loc[:,'logDiff'].rolling(5).sum()
    tickerDF[ticker].loc[:,'fwd5'] = tickerDF[ticker].loc[:,'logDiff'].rolling(5).sum().shift(-5)
    tickerDF[ticker].loc[:,'fwdVol5'] = tickerDF[ticker].loc[:,'logDiff'].rolling(5).std().shift(-5)
    tickerDF[ticker].loc[:,'fwd15'] = tickerDF[ticker].loc[:,'logDiff'].rolling(15).sum().shift(-15)
    tickerDF[ticker].loc[:,'fwdVol15'] = tickerDF[ticker].loc[:,'logDiff'].rolling(15).std().shift(-15)
    #drawdownCutoff[ticker] = -tickerDF[ticker].loc[:,'logDiff'].std()*drawdownMultiple
    #tickerDF[ticker].loc[:,'drawdownTrigger'] = tickerDF[ticker].loc[:,'Close'] * (1+drawdownCutoff[ticker])
    #tickerDF[ticker].loc[:,'drawdown'] = tickerDF[ticker].loc[:,'Low'].rolling(15).min().shift(-15) < tickerDF[ticker].loc[:,'drawdownTrigger']
    averageChange[ticker] = tickerDF[ticker].loc[:,'logDiff'].mean()
    tickerDF[ticker].loc[:,'cumStd'] = tickerDF[ticker].loc[:,'logDiff'].expanding().std()
    tickerDF[ticker].loc[:,'cumMean'] = tickerDF[ticker].loc[:,'logDiff'].expanding().mean()
    tickerDF[ticker].loc[:,'demeanChange'] = tickerDF[ticker].loc[:,'logDiff']-tickerDF[ticker].loc[:,'cumMean']
    tickerDF[ticker].loc[:,'demeanCumChange'] = tickerDF[ticker].loc[:,'demeanChange'].expanding().sum()
    tickerDF[ticker].loc[:,'minChange'] = tickerDF[ticker].loc[:,'demeanCumChange'].expanding().min()
    tickerDF[ticker].loc[:,'maxChange'] = tickerDF[ticker].loc[:,'demeanCumChange'].expanding().max()
    tickerDF[ticker].loc[:,'chgRange'] = tickerDF[ticker].loc[:,'maxChange']-tickerDF[ticker].loc[:,'minChange']
    tickerDF[ticker].loc[:,'chgRatio'] = tickerDF[ticker].loc[:,'chgRange']/tickerDF[ticker].loc[:,'cumStd']

    hurstMod[ticker] = RollingOLS(np.log(tickerDF[ticker].loc[:,'chgRatio']).dropna(), sm.add_constant(np.log(range(1,1+len(tickerDF[ticker].loc[:,'chgRatio'].dropna()))), prepend=False), window=300)
    recursiveHurst[ticker] = sm.RecursiveLS(np.log(tickerDF[ticker].loc[:,'chgRatio']).dropna(), sm.add_constant(np.log(range(1,1+len(tickerDF[ticker].loc[:,'chgRatio'].dropna()))), prepend=False))
    res=hurstMod[ticker].fit()
    res2 = recursiveHurst[ticker].fit()
    res2.plot_recursive_coefficient()
    #plt.figure()
    #sns.regplot(np.log(range(1,1+len(tickerDF[ticker].loc[:,'chgRatio'].dropna()))), np.log(tickerDF[ticker].loc[:,'chgRatio']).dropna()).set_title(ticker)
    #sns.lineplot(tickerDF[ticker].loc[:,'chgRatio'].dropna().index, res.params.loc[:,'x1'])

    #plotTicker='IEF'
    #sns.lineplot(x=tickerDF[ticker].index, y=tickerDF[ticker].loc[:,'chgRatio'])


#fractional Differentiation    
ticker='SPY'
slow_speed = .4
slow_window = 900
tickerDF[ticker].loc[:,'LogPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
[slow_FD, adf_res] = fracDiff(tickerDF['SPY'],'Close', slow_speed, slow_window, 250)
print('slow speed ADF p-stat: ' + str(adf_res))
plt.plot(slow_FD.rolling(30).mean())

#fractional Differentiation    
ticker='SPY'
medium_speed = .625
medium_window = 400
tickerDF[ticker].loc[:,'LogPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
[medium_FD, adf_res_medium] = fracDiff(tickerDF['SPY'],'LogPx', medium_speed, medium_window, 100)
print('slow speed ADF p-stat: ' + str(adf_res_medium))
medium_signal = (medium_FD-medium_FD.mean())#/medium_FD.std()
plt.plot(medium_FD.iloc[-500:])
plt.plot(medium_FD.rolling(30).mean())





#calculate and store fractional dimensions or just load them    
#dimensions_long_term = scanAndStore(800, dimensionFile_long_term)
#dimensions_month = scanAndStore(30, 'dimensions_month.obj')
dimensions_month = loadDimensions('dimensions_month.obj')
dimensions_long_term = loadDimensions('dimensions_long_term.obj')


squareSize = int(np.ceil(np.sqrt(len(test_stocks))))
sns.set(style="darkgrid")
fig = plt.figure(figsize=(9,9), dpi=300)
for i, ticker in enumerate(test_stocks):
    input_table = (frac.fracDiff(tickerDF[ticker], 'logPx', dimensions_month[1][ticker], 30, 200)[0])[-200:]
    plot_data = (input_table - input_table.mean())/input_table.std()
    ax = plt.subplot(squareSize,squareSize,i+1)
    ax.title.set_text(ticker)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)
    sns.lineplot(data=input_table,  color='xkcd:jade')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])    


    




# temp_data = pd.Series()
# weights = fractionalWeights(cutoffWindowDaily,fraction)
# tickerDF[ticker].loc[:,'LogPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
# tickerDF[ticker].loc[:,'LogPx'].iloc[-1*cutoffWindowDaily:].dot(weights)
# for i in range (cutoffWindowDaily,tickerDF[ticker].shape[0]):
#     temp_data[tickerDF[ticker].index[i]] = (tickerDF[ticker].loc[:,'LogPx'].iloc[-1*cutoffWindowDaily+i:i]).dot(weights)
#     #print(-1*cutoffWindowDaily+i)
#     print(i)

# temp_data = temp_data - temp_data.mean()
# integrated_data = pd.Series()
# weights = fractionalWeights(cutoffWindowDaily,-1*fraction)
# for i in range (cutoffWindowDaily, temp_data.shape[0]):
#     integrated_data[tickerDF[ticker].index[i]] = (temp_data.iloc[i-cutoffWindowDaily:i]).dot(weights)
#     #print(-1*cutoffWindowDaily+i)
#     print(i)
# result = adfuller(temp_data)
# print(result[1])
# plt.plot(temp_data) 


# fracDiff(tickerDF['SPY'],'LogPx', .42, 900, 50)

# # #any use to this stuff?
# # ticker='XLU'
# # tickerDF[ticker].loc[:,'LogPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
# # plt.plot(tickerDF[ticker].loc[:,'Volume'].cumsum(),tickerDF[ticker].loc[:,'LogPx'])
# # plt.plot(tickerDF[ticker].loc[:,'LogPx'])
# # plt.plot(tickerDF[ticker].loc[:,'Volume'].cumsum())
# ticker='BSV'
# nolds.hurst_rs(tickerDF[ticker].loc[:,'Close'].values, debug_data=True, debug_plot=True)

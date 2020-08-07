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
-
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

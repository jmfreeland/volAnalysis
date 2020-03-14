# -*- coding: utf-8 -*-
"""
Portfolio Analysis
todo: adjust for dividends

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

#Pick stocks for analysis
test_stocks = ['SPY', 'TLT', 'IEF','TOTL','BSV','XLU','XLP','DEF']
tickerData = {}
tickerDF = {}
drawdownCutoff = {}


drawdownMultiple = 10

for ticker in test_stocks:
    tickerData[ticker] = yf.Ticker(ticker)
    #get the historical prices for this ticker
    tickerDF[ticker] = tickerData[ticker].history(period='1d', start='1980-1-1', end='2020-12-31')
 

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
    drawdownCutoff[ticker] = -tickerDF[ticker].loc[:,'logDiff'].std()*drawdownMultiple
    tickerDF[ticker].loc[:,'drawdownTrigger'] = tickerDF[ticker].loc[:,'Close'] * (1+drawdownCutoff[ticker])
    tickerDF[ticker].loc[:,'drawdown'] = tickerDF[ticker].loc[:,'Low'].rolling(15).min().shift(-15) < tickerDF[ticker].loc[:,'drawdownTrigger']



squareSize = int(np.ceil(np.sqrt(len(test_stocks))))

sns.set(style="darkgrid")

fig = plt.figure(figsize=(9,9), dpi=300)
fig.suptitle(' 5/15/30-day (blue,purple,green) Trailing Vol vs. 15-day Fwd. Vol')
for i, ticker in enumerate(test_stocks):
    inputTable = tickerDF[ticker].dropna()
    ax = plt.subplot(squareSize,squareSize,i+1)
    ax.title.set_text(ticker)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)
    sns.regplot(x=inputTable.loc[:,'vol30'], y=inputTable.loc[:,'fwdVol15'],order=2, marker='o', color='xkcd:jade' ,scatter_kws={'s':1,'alpha':.1})
    sns.regplot(x=inputTable.loc[:,'vol15'], y=inputTable.loc[:,'fwdVol15'],order=2, marker='o', color='xkcd:amethyst', scatter_kws={'s':1,'alpha':.1})
    sns.regplot(x=inputTable.loc[:,'vol5'], y=inputTable.loc[:,'fwdVol15'],order=2, marker='o', color='xkcd:deep blue', scatter_kws={'s':1,'alpha':.1})
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


fig = plt.figure(figsize=(9,9), dpi=300)
fig.suptitle(' 5/15/30-day (blue,purple,green) Trailing Vol vs. 15-day Fwd. Return')
for i, ticker in enumerate(test_stocks):
    inputTable = tickerDF[ticker].dropna()
    ax = plt.subplot(squareSize,squareSize,i+1)
    ax.title.set_text(ticker)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)
    sns.regplot(x=inputTable.loc[:,'vol30'], y=inputTable.loc[:,'fwd15'],order=2, marker='o', color='xkcd:jade' ,scatter_kws={'s':1,'alpha':.1})
    sns.regplot(x=inputTable.loc[:,'vol15'], y=inputTable.loc[:,'fwd15'],order=2, marker='o', color='xkcd:amethyst', scatter_kws={'s':1,'alpha':.1})
    sns.regplot(x=inputTable.loc[:,'vol5'], y=inputTable.loc[:,'fwd15'],order=2, marker='o', color='xkcd:deep blue', scatter_kws={'s':1,'alpha':.1})
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


fig = plt.figure(figsize=(9,9), dpi=300)
fig.suptitle(' 5/15/30-day (blue,purple,green) Trailing Vol vs. 15-day Fwd. Return')
for i, ticker in enumerate(test_stocks):
    inputTable = tickerDF[ticker].dropna()
    ax = plt.subplot(squareSize,squareSize,i+1)
    ax.title.set_text(ticker)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)
    sns.regplot(x=inputTable.loc[:,'vol30'], y=inputTable.loc[:,'drawdown'],order=2, marker='o', color='xkcd:jade' ,scatter_kws={'s':1,'alpha':.1})
    sns.regplot(x=inputTable.loc[:,'vol15'], y=inputTable.loc[:,'drawdown'],order=2, marker='o', color='xkcd:amethyst', scatter_kws={'s':1,'alpha':.1})
    sns.regplot(x=inputTable.loc[:,'vol5'], y=inputTable.loc[:,'drawdown'],order=2, marker='o', color='xkcd:deep blue', scatter_kws={'s':1,'alpha':.1})
fig.tight_layout(rect=[0, 0.03, 1, 0.95])




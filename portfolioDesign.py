# -*- coding: utf-8 -*-
"""
Portfolio Construction:
Objective: Calculate approximate forward return and scale accepted risk up or down accordingly.
inputs: FRED data on spreads and total returns, MOVE Index, yield curve, VIX and returns
outputs: Location of current spreads in historical range, forward return 
            projections, spread change projections, volatility vs. volume/autocorrelation comparison
            -PCA Analysis
ToDo: -Organize forward return model predictions into useful output
      -Calculate return estimates for spreads and yields
      -Consider other estimates like worst drawdown etc.
    
    
"""

import pandas as pd
import numpy as np
#import quandl
import seaborn as sns
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm

#initializations
sns.set()

#pull Investment Grade spreads, total return data from FRED 
start = dt.datetime(1996, 1, 1)
end = dt.date.today()
IG_ICE_data = pdr.get_data_fred(['BAMLC0A0CM','BAMLC0A4CBBB','BAMLC8A0C15PY','BAMLC0A0CMEY'], start, end)

fig = plt.figure(figsize=[8,8])
plt.title("ICE BofA IG Spread Histogram, 1996-")
sns.distplot(IG_ICE_data.loc[:,'BAMLC0A0CM'], bins=50, label='IG', hist_kws={"alpha": .7})
sns.distplot(IG_ICE_data.loc[:,'BAMLC0A4CBBB'], bins=50, label='BBB', hist_kws={"alpha": .1})
sns.distplot(IG_ICE_data.loc[:,'BAMLC8A0C15PY'], bins=50, label='15Y+', color='purple', hist_kws={"alpha": .1})

plt.xlabel('IG (blue), BBB (orange), 15yt+ (purple)')
IG_ICE_quantile = IG_ICE_data.loc[:,'BAMLC0A0CM'].rank(pct=True)[-1]
BBB_ICE_quantile = IG_ICE_data.loc[:,'BAMLC0A4CBBB'].rank(pct=True)[-1]
Long_ICE_quantile = IG_ICE_data.loc[:,'BAMLC8A0C15PY'].rank(pct=True)[-1]

print("ICE/BofA IG spreads of " + str(100*IG_ICE_data.loc[:,'BAMLC0A0CM'][-1]) + "bp are higher than " + '{:.1%}'.format(IG_ICE_quantile) + " of history (1919-current)")
print("ICE/BofA BBB spreads of " + str(100*IG_ICE_data.loc[:,'BAMLC0A4CBBB'][-1]) + "bp are higher than " + '{:.1%}'.format(BBB_ICE_quantile) + " of history (1919-current)")
print("ICE/BofA 15yr+ spreads of " + str(100*IG_ICE_data.loc[:,'BAMLC8A0C15PY'][-1]) + "bp are higher than " + '{:.1%}'.format(Long_ICE_quantile) + " of history (1919-current)")

#print("1yr. forward yield change from this level is typically: ")

start = dt.datetime(1919, 1, 1)
end = dt.date.today() 
IG_moodys_data = pdr.get_data_fred(['BAA','AAA'], start, end)
IG_moodys_daily = pdr.get_data_fred(['DBAA','DAAA'], end + dt.timedelta(days=-7), end)
IG_moodys_data.loc[IG_moodys_daily.index[-1],'BAA'] = IG_moodys_daily.loc[:,'DBAA'][-1]
IG_moodys_data.loc[IG_moodys_daily.index[-1],'AAA'] = IG_moodys_daily.loc[:,'DAAA'][-1]

fig = plt.figure(figsize=[8,8])
plt.title("Moody's Baa/Aaa Yield Histogram, 1919-")
sns.distplot(IG_moodys_data.loc[:,'BAA'], bins=50, label='Baa')
sns.distplot(IG_moodys_data.loc[:,'AAA'], bins=50, label='Aaa')
plt.xlabel('Baa (blue), Aaa (orange)')
Baa_quantile = IG_moodys_data.loc[:,'BAA'].rank(pct=True)[-1]
Aaa_quantile = IG_moodys_data.loc[:,'AAA'].rank(pct=True)[-1]

print("Moody's Baa yields of " + '{:.1%}'.format(.01*IG_moodys_data.loc[:,'BAA'][-1]) + " are higher than " + '{:.1%}'.format(Baa_quantile) + " of history (1919-current)")
print("Moody's Aaa yields of " + '{:.1%}'.format(.01*IG_moodys_data.loc[:,'AAA'][-1]) + " are higher than " + '{:.1%}'.format(Aaa_quantile) + " of history (1919-current)")
#print("1yr. forward yield change from this level is typically: ")


#Begin Equity Analysis
FRED_equity_data = pdr.get_data_fred(['NCBEILQ027S','BCNSDODNS','CMDEBT','FGSDODNS','SLGSDODNS','FBCELLQ027S','DODFFSWCMI'], start, end)
#(a+f)/1000)/(((a+f)/1000)+b+c+d+e+g)
equity_allocation = ((FRED_equity_data.loc[:,'NCBEILQ027S']+FRED_equity_data.loc[:,'FBCELLQ027S'])/1000)/(((FRED_equity_data.loc[:,'NCBEILQ027S']+FRED_equity_data.loc[:,'FBCELLQ027S'])/1000)+FRED_equity_data.loc[:,'BCNSDODNS']+FRED_equity_data.loc[:,'CMDEBT']+FRED_equity_data.loc[:,'FGSDODNS']+FRED_equity_data.loc[:,'SLGSDODNS']+FRED_equity_data.loc[:,'DODFFSWCMI'])

tickerData = {}
tickerDF = {}
prediction_data = {}
model_outputs = {}
model_stats = {}

test_assets = ['^SP500TR','^GSPC'] #,'^XNDX','D1AR.DE']
for ticker in test_assets:
    tickerData[ticker] = yf.Ticker(ticker)
    #get the historical prices for this ticker
    tickerDF[ticker] = tickerData[ticker].history(period='1w', start='1971-1-1', end='2020-12-31')
    prediction_data[ticker] = pd.merge_asof(left=pd.DataFrame(equity_allocation, columns=['equity_allocation']), right=tickerDF[ticker], left_index=True, right_index=True).dropna()
    prediction_data[ticker].loc[:,'logDiff'] = np.log(prediction_data[ticker].loc[:,'Close']).diff()
    prediction_data[ticker].loc[:,'fwd1'] = prediction_data[ticker].loc[:,'logDiff'].rolling(4).sum().shift(-4)
    prediction_data[ticker].loc[:,'fwd3'] = prediction_data[ticker].loc[:,'logDiff'].rolling(12).sum().shift(-12)
    prediction_data[ticker].loc[:,'fwd5'] = prediction_data[ticker].loc[:,'logDiff'].rolling(20).sum().shift(-20)
    prediction_data[ticker].loc[:,'fwd10'] = prediction_data[ticker].loc[:,'logDiff'].rolling(40).sum().shift(-40)
    
    
    model_outputs[ticker] = {}
    model_stats[ticker] = {}

    current_allocation = equity_allocation[-1]
    allocation_quantile = prediction_data[ticker].loc[:,'equity_allocation'].rank(pct=True)[-1]    
    print('\n*Equity allocation is higher than ' + '{:.1%}'.format(allocation_quantile) + ' of available data points for ' + ticker + '.')
       

    print('\nEquity Allocation vs. Forward Returns:')
    tmp_data = prediction_data[ticker].loc[:,['equity_allocation','fwd1']].dropna()
    X = sm.add_constant(tmp_data.loc[:,'equity_allocation'])
    Y = tmp_data.loc[:,'fwd1']
    model = sm.OLS(Y,X)
    results = model.fit()
    print("Equity Allocation of " + '{:.1%}'.format(current_allocation) + " implies a 1yr return of " + '{:.1%}'.format(results.predict([1, current_allocation])[0]) + ' for ' + ticker)
    (model_stats[ticker])['equityAlloc_1yrFwd'] = results
    fig = plt.figure(figsize=(9,9), dpi=300)
    sns.regplot(x=X.iloc[:,1],y=Y)
    fig.suptitle('1yr. Forward Return [' + ticker + '] vs. Equity Allocation')  

    
    tmp_data = prediction_data[ticker].loc[:,['equity_allocation','fwd3']].dropna()
    X = sm.add_constant(tmp_data.loc[:,'equity_allocation'])
    Y = tmp_data.loc[:,'fwd3']
    model = sm.OLS(Y,X)
    results = model.fit()
    print("Equity Allocation of " + '{:.1%}'.format(current_allocation) + " implies a 3yr return of " + '{:.1%}'.format(results.predict([1, current_allocation])[0]) + ' for ' + ticker)
    (model_stats[ticker])['equityAlloc_3yrFwd'] = results
    fig = plt.figure(figsize=(9,9), dpi=300)
    sns.regplot(x=X.iloc[:,1],y=Y)
    fig.suptitle('3yr. Forward Return [' + ticker + '] vs. Equity Allocation')    
    
    
    tmp_data = prediction_data[ticker].loc[:,['equity_allocation','fwd5']].dropna()
    X = sm.add_constant(tmp_data.loc[:,'equity_allocation'])
    Y = tmp_data.loc[:,'fwd5']
    model = sm.OLS(Y,X)
    results = model.fit()
    print("Equity Allocation of " + '{:.1%}'.format(current_allocation) + " implies a 5yr return of " + '{:.1%}'.format(results.predict([1, current_allocation])[0]) + ' for ' + ticker)
    (model_stats[ticker])['equityAlloc_5yrFwd'] = results
    fig = plt.figure(figsize=(9,9), dpi=300)
    sns.regplot(x=X.iloc[:,1],y=Y)
    fig.suptitle('5yr. Forward Return [' + ticker + '] vs. Equity Allocation')    
    
    
    tmp_data = prediction_data[ticker].loc[:,['equity_allocation','fwd10']].dropna()
    X = sm.add_constant(tmp_data.loc[:,'equity_allocation'])
    Y = tmp_data.loc[:,'fwd10']
    model = sm.OLS(Y,X)
    results = model.fit()
    print("Equity Allocation of " + '{:.1%}'.format(current_allocation) + " implies a 10yr return of " + '{:.1%}'.format(results.predict([1, current_allocation])[0]) + ' for ' + ticker)
    (model_stats[ticker])['equityAlloc_10yrFwd'] = results
    fig = plt.figure(figsize=(9,9), dpi=300)
    sns.regplot(x=X.iloc[:,1],y=Y)
    fig.suptitle('10yr. Forward Return [' + ticker + '] vs. Equity Allocation')  
    
    
    



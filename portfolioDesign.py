# -*- coding: utf-8 -*-
"""
Portfolio Construction:
Objective: Calculate approximate forward return and scale accepted risk up or down accordingly.
inputs: FRED data on spreads and total returns, MOVE Index, yield curve, VIX and returns
outputs: Location of current spreads in historical range, forward return 
            projections, spread change projections
    
"""

import pandas as pd
import numpy as np
#import quandl
import seaborn as sns
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt

#initializations
sns.set()

#pull Investment Grade spreads, total return data from FRED 
start = dt.datetime(1996, 1, 1)
end = dt.date.today()
IG_ICE_data = pdr.get_data_fred(['BAMLC0A0CM','BAMLC0A4CBBB','BAMLC0A0CMEY'], start, end)

fig = plt.figure(figsize=[8,8])
plt.title("ICE BofA IG Spread Histogram, 1996-")
sns.distplot(IG_ICE_data.loc[:,'BAMLC0A0CM'], bins=50, label='IG')
sns.distplot(IG_ICE_data.loc[:,'BAMLC0A4CBBB'], bins=50, label='BBB')
plt.xlabel('IG (blue), BBB (orange)')
IG_ICE_quantile = IG_ICE_data.loc[:,'BAMLC0A0CM'].rank(pct=True)[-1]
BBB_ICE_quantile = IG_ICE_data.loc[:,'BAMLC0A4CBBB'].rank(pct=True)[-1]

print("ICE/BofA IG spreads of " + str(100*IG_ICE_data.loc[:,'BAMLC0A0CM'][-1]) + "bp are higher than " + '{:.1%}'.format(IG_ICE_quantile) + " of history (1919-current)")
print("ICE/BofA BBB spreads of " + str(100*IG_ICE_data.loc[:,'BAMLC0A4CBBB'][-1]) + "bp are higher than " + '{:.1%}'.format(BBB_ICE_quantile) + " of history (1919-current)")
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


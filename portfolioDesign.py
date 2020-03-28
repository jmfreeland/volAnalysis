# -*- coding: utf-8 -*-
"""
Portfolio Construction:
Calculate approximate forward return and scale accepted risk up or down
inputs: FRED data on spreads and total returns, MOVE Index, yield curve, VIX and returns
outputs: 
    
"""

import pandas as pd
import numpy as np
#import quandl
import seaborn as sns
import pandas_datareader as pdr
import datetime as dt

#initializations
sns.set()

#pull Investment Grade spreads, total return data from FRED 
start = dt.datetime(1996, 1, 1)
end = dt.date.today()
IG_ICE_data = pdr.get_data_fred(['BAMLC0A0CM','BAMLC0A4CBBB','BAMLC0A0CMEY'], start, end)

start = dt.datetime(1919, 1, 1)
end = dt.date.today() 
IG_moodys_data = pdr.get_data_fred(['BAA','AAA'], start, end)
IG_moodys_daily = pdr.get_data_fred(['DBAA','DAAA'], end + dt.timedelta(days=-7), end)


print("Moody's BBB yields are in the " + " percentile")
print("1yr. forward yield change from this level is typically: ")

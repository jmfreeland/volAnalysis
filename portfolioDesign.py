# -*- coding: utf-8 -*-
"""
Portfolio Construction:
Objective: Calculate approximate forward return and scale accepted risk up or down accordingly.
Questions:  What are expected returns for equity risk?
            What are expected fixed income returns?
            How much can they be expected to deviate?
            


inputs: FRED data on spreads and total returns, MOVE Index, yield curve, VIX and returns
outputs: Location of current spreads in historical range, forward return 
            projections, spread change projections, volatility vs. volume/autocorrelation comparison
            -PCA Analysis
ToDo: -Improve length of data
      -Calculate return estimates for spreads and yields
      -Consider other estimates like worst drawdown etc.
      -Organize into subplots into a 1x4
      -How much can you speed up FRED estimator?
      -Where can you maximize sharpe in fixed income?
      -What is forward volatility estimate based on model?
      -Calculate highest sharpe fixed income
      -fix carry for multiple days between dates
      -Calculate implied vol premium/discount
      -Pull in schiller CAPE
      -Rolling vol analysis (rates, stocks, etc)
      -Correlate rates to price vol
      -Simon ward Switching models
      -Simon Ward Portfolio theory model
      -Real Rates / Breakevens
      -Position Sizing
    
Rationale: Shoot for a target Sharpe ratio of 2. Accept net long risk in proportion to estimated
            returns available. As estimated return increases (assets are cheap), increase 
            volatility target to allow for 2.0 sharpe. 
            
            
    
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
import QuantLib as ql

#function definitions
def risk_budget(account_value):  
    return [vol_target*account_value, vol_target*account_value*np.sqrt(252)]

#function definitions
def annual_return_equities():
    return annual_return


#initializations
sns.set()


#Get current risk free rate
risk_free = pdr.get_data_fred(['USD3MTD156N'], dt.datetime(2020,1,1), dt.date.today())
r = (risk_free.loc[:,'USD3MTD156N'].values)[-1]
target_sharpe = 2


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
equity_allocation = ((FRED_equity_data.loc[:,'NCBEILQ027S']+FRED_equity_data.loc[:,'FBCELLQ027S'])/1000)/(((FRED_equity_data.loc[:,'NCBEILQ027S']+FRED_equity_data.loc[:,'FBCELLQ027S'])/1000)+FRED_equity_data.loc[:,'BCNSDODNS']+FRED_equity_data.loc[:,'CMDEBT']+FRED_equity_data.loc[:,'FGSDODNS']+FRED_equity_data.loc[:,'SLGSDODNS']+FRED_equity_data.loc[:,'DODFFSWCMI'])
FRED_end_date = FRED_equity_data.index[-1] + dt.timedelta(days=90)

tickerData = {}
tickerDF = {}
prediction_data = {}
model_outputs = {}
model_stats = {}

test_assets = ['^SP500TR','^GSPC','^IXIC'] #,'^XNDX','D1AR.DE']
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

    #current_allocation = equity_allocation[-1]
    print('Implying current allocation from S&P 500 Total Returns')
    post_report = tickerDF['^SP500TR'].loc[tickerDF['^SP500TR'].index>FRED_end_date,'Close']
    post_report_return = post_report[-1]/post_report[0]
    last_report = FRED_equity_data.iloc[-1] 
    current_allocation = ((last_report['NCBEILQ027S']*post_report_return+last_report['FBCELLQ027S']*post_report_return)/1000)/(((last_report['NCBEILQ027S']*post_report_return+last_report['FBCELLQ027S']*post_report_return)/1000)+last_report['BCNSDODNS']+last_report['CMDEBT']+last_report['FGSDODNS']+last_report['SLGSDODNS']+last_report['DODFFSWCMI'])

    
    allocation_quantile = prediction_data[ticker].loc[:,'equity_allocation'].append(pd.Series(current_allocation)).rank(pct=True).iloc[-1]    
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
    
    
tenYr_return = model_stats['^SP500TR']['equityAlloc_10yrFwd'].predict([1, current_allocation])[0]
annual_return = np.log(1+tenYr_return)/10
excess_return_prediction = annual_return - r/100
vol_target = excess_return_prediction / target_sharpe
    
print('\n10yr forward return of S&P 500 annualizes to ' + '{:.1%}'.format(annual_return))
print('Risk free rate is ' + '{:.2%}'.format(r/100))
print('Excess return predictions is ' + '{:.2%}'.format(excess_return_prediction))
print('Annual vol target for ' + str(target_sharpe) + ' sharpe is: ' + '{:.2%}'.format(vol_target))


#Begin Interest Rate Analysis
bond_yields = pdr.get_data_fred(['DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30'], dt.datetime(1962,1,1), dt.datetime(2020,2,1))
maturities = [(2,'DGS2'),(3,'DGS3'),(5,'DGS5'),(7,'DGS7'),(10,'DGS10'),(20,'DGS20'),(30,'DGS30')]

duration = pd.DataFrame()
# 1/bond_yields.loc[:,'DGS2'] * (1-1/(1+.5*bond_yields.loc[:,'DGS2'])**2*2)
#ond_yields = bond_yields.dropna()
#need to drop non-sequential days

dayCount = ql.Thirty360()
calendar = ql.UnitedStates()
interpolation = ql.Linear()
compounding = ql.Compounded
compoundingFrequency = ql.Annual

curve_point = (2,'DGS2')
test_yields = bond_yields.loc[:,curve_point[1]].dropna()
valid_indices = np.where((np.diff((test_yields.index).values).astype('timedelta64[D]').astype(int))<7)[0]
valid_dates = test_yields.iloc[valid_indices]


return_analysis = {}
for curve_point in maturities:
    test_yields = bond_yields.loc[:,curve_point[1]].dropna()
    valid_indices = np.where((np.diff((test_yields.index).values).astype('timedelta64[D]').astype(int))<7)[0]
    valid_dates = test_yields.iloc[valid_indices]
    
    return_analysis[curve_point] = pd.DataFrame()
    for idx, eval_date in enumerate(valid_dates.index):
        start = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        maturity = start+365*curve_point[0]
        schedule = ql.MakeSchedule(start, maturity, ql.Period('6M'))
        interest = ql.FixedRateLeg(schedule, ql.Actual365Fixed(), [100.], [test_yields.loc[eval_date]/100])
        bond = ql.Bond(0, ql.UnitedStates(), start, interest)
        rate = ql.InterestRate(test_yields.loc[eval_date]/100, ql.Actual365Fixed(), ql.Simple, ql.Annual)
        open_price = ql.BondFunctions.cleanPrice(bond,rate,start)
        next_location = valid_indices[idx]+1
        rate_close = ql.InterestRate(test_yields.iloc[next_location]/100, ql.Actual365Fixed(), ql.Simple, ql.Annual)
        close_price = ql.BondFunctions.cleanPrice(bond,rate_close,start+1)
        price_return = close_price-open_price
        carry = (test_yields.loc[eval_date] / 365)
        total_return = price_return + carry
        return_analysis[curve_point].loc[eval_date,'open_yield'] = test_yields.loc[eval_date]/100
        return_analysis[curve_point].loc[eval_date,'close_yield'] = test_yields.iloc[next_location]/100
        return_analysis[curve_point].loc[eval_date,'open_price'] = open_price
        return_analysis[curve_point].loc[eval_date,'close_price'] = close_price
        return_analysis[curve_point].loc[eval_date,'price_return'] = price_return
        return_analysis[curve_point].loc[eval_date,'carry'] = carry
        return_analysis[curve_point].loc[eval_date,'total_return'] = total_return
        return_analysis[curve_point].loc[eval_date,'abs_move'] = abs(total_return)
        

#graph analyses - 2y
fig = plt.figure(figsize=(9,9), dpi=300)
sns.lineplot(data=return_analysis[(2,'DGS2')].loc[:,'total_return'])
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(2,'DGS2')], x='open_yield', y='total_return', fit_reg=True)
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(2,'DGS2')], x='open_yield', y='abs_move', fit_reg=True)

#graph analyses - 5y
fig = plt.figure(figsize=(9,9), dpi=300)
sns.lineplot(data=return_analysis[(5,'DGS5')].loc[:,'total_return'])
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(5,'DGS5')], x='open_yield', y='total_return', fit_reg=True)
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(5,'DGS5')], x='open_yield', y='abs_move', fit_reg=True)

#graph analyses - 7y
fig = plt.figure(figsize=(9,9), dpi=300)
sns.lineplot(data=return_analysis[(7,'DGS7')].loc[:,'total_return'])
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(7,'DGS7')], x='open_yield', y='total_return', fit_reg=True)
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(7,'DGS7')], x='open_yield', y='abs_move', fit_reg=True)

#graph analyses - 10y
fig = plt.figure(figsize=(9,9), dpi=300)
sns.lineplot(data=return_analysis[(10,'DGS10')].loc[:,'total_return'])
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(10,'DGS10')], x='open_yield', y='total_return', fit_reg=True)
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(10,'DGS10')], x='open_yield', y='abs_move', fit_reg=True)

#graph analyses - 20y
fig = plt.figure(figsize=(9,9), dpi=300)
sns.lineplot(data=return_analysis[(20,'DGS20')].loc[:,'total_return'])
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(20,'DGS20')], x='open_yield', y='total_return', fit_reg=True)
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(20,'DGS20')], x='open_yield', y='abs_move', fit_reg=True)

#graph analyses - 30y
fig = plt.figure(figsize=(9,9), dpi=300)
sns.lineplot(data=return_analysis[(30,'DGS30')].loc[:,'total_return'])
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(30,'DGS30')], x='open_yield', y='total_return', fit_reg=True)
fig = plt.figure(figsize=(9,9), dpi=300)
sns.regplot(data=return_analysis[(30,'DGS30')], x='open_yield', y='abs_move', fit_reg=True)


#calculate return & vol stats [not entirely comparable]
return_2y = (1+return_analysis[(2,'DGS2')].loc[:,'total_return']/100).product()-1
return_5y = (1+return_analysis[(5,'DGS5')].loc[:,'total_return']/100).product()-1
daily_vol_2y = (return_analysis[(2,'DGS2')].loc[:,'total_return']/100).std()
daily_vol_5y = (return_analysis[(5,'DGS5')].loc[:,'total_return']/100).std()
daily_vol_10y = (return_analysis[(10,'DGS10')].loc[:,'total_return']/100).std()
mean_return_2y = (return_analysis[(2,'DGS2')].loc[:,'total_return']/100).mean()

daily_vol_2y*np.sqrt(252)
daily_vol_5y*np.sqrt(252)
daily_vol_10y*np.sqrt(252)


# #diagnostics
# ql.BondFunctions.startDate(bond)
# ql.BondFunctions.maturityDate(bond)
# test_yields[eval_date]/100
# ql.BondFunctions.nextCashFlowDate(bond, start)
# ql.BondFunctions.nextCashFlowAmount(bond, start)


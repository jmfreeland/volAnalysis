from scipy.optimize import minimize
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

#attempt to optimize lookback, there appeared to be an improvement around 1100


# def fracDiffTest(fraction, cutoffWindowDaily):
#     ticker='SPY'
#     #fracDiff[ticker] = {}
#     temp_data = pd.Series()
#     weights = fractionalWeights(cutoffWindowDaily,fraction)
#     tickerDF[ticker].loc[:,'LogPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
#     tickerDF[ticker].loc[:,'LogPx'].iloc[-1*cutoffWindowDaily:].dot(weights)
#     for i in range (cutoffWindowDaily,tickerDF[ticker].shape[0]):
#         temp_data[tickerDF[ticker].index[i]] = (tickerDF[ticker].loc[:,'LogPx'].iloc[-1*cutoffWindowDaily+i:i]).dot(weights)
#         #print(-1*cutoffWindowDaily+i)
#     #print(i)
#     result = adfuller(temp_data, autolag='AIC')
#     return result[1]


def fractionalWeights(k, d):
    w = np.array([1.0], dtype=float)
    for i in range(1,k):
        w = np.append(w, -1*w[i-1]*((d-i+1)/i))
    print(w.sum())
    return w

#sum asymptotically approaches 0. It is 1 + series of negative numbers
def fractionalWeightsCutoff(d, cutoff):
    w = np.array([1.0], dtype=float)
    i=1
    while (w.sum()>cutoff):
        w = np.append(w, -1*w[i-1]*((d-i+1)/i))
        i+=1
        print(w.sum())
    return w

def fracDiff(input_data, test_col, fraction, cutoffWindow, afdMaxLag):
    temp_data = pd.Series()
    weights = fractionalWeights(cutoffWindow,fraction)
    for i in range (cutoffWindow,input_data.shape[0]):
        temp_data[input_data.index[i]] = (input_data.loc[:,test_col].iloc[-1*cutoffWindow+i:i]).dot(np.flipud(weights))
        #print(-1*cutoffWindowDaily+i)
    #debug code
    #print(i)3
    #print(str(input_data.loc[:,test_col].iloc[-1*cutoffWindow+1:i]))
    #print(np.flipud(weights))
    result = adfuller(temp_data, autolag='AIC')
    return temp_data, result[1]

def fracDiffDebug(input_data, test_col, fraction, cutoffWindow, afdMaxLag, point):
    temp_data = pd.Series()
    weights = fractionalWeights(cutoffWindow,fraction)
    for i in range (cutoffWindow,input_data.shape[0]):
        temp_data[input_data.index[i]] = (input_data.loc[:,test_col].iloc[-1*cutoffWindow+i:i]).dot(np.flipud(weights))
        print(str(i))
    #debug code
    #print(i)
    #print(str(input_data.loc[:,test_col].iloc[-1*cutoffWindow+1:i]))
    #print(np.flipud(weights))
    result = adfuller(temp_data, autolag='AIC')
    return temp_data[point], (input_data.loc[:,test_col].iloc[point+1:cutoffWindow+1+point]), np.flipud(weights) 


def fracDiffOpt(input_data, test_col, fixed_window, starting_dim, increments, cutoff, max_lags ):
    res = 0
    i=0
    dim=np.array([starting_dim])
    while res<cutoff and dim[i] >=.01:
        [tmp, res] = fracDiff(input_data, test_col, dim[i], fixed_window, max_lags)
        dim = np.append(dim, dim[i] - starting_dim/increments)
        print(dim[i])
        i+=1
    return dim[i-2]

#should separate out fallback bit
def fracDiffScan(input_data, test_col, test_stocks, fixed_window, starting_dim, increments, cutoff, max_lags):
    dimension_list = {}
    dimension_list_mod = {}
    for ticker in test_stocks:
        dimension_list[ticker] = fracDiffOpt(input_data[ticker], test_col, fixed_window, starting_dim, increments, cutoff, max_lags)
     
    for ticker in test_stocks:
        if input_data[ticker].shape[0]<2500:
            dimension_list_mod[ticker] = dimension_list['SPY']
        else:
            dimension_list_mod[ticker] = dimension_list[ticker]
    return dimension_list, dimension_list_mod
        


 
# #test window performance
# ticker='SPY'
# window_perf = pd.Series()
# for i in range (1,50):
#    [tmp, window_perf.loc[str(i)]] = fracDiff(tickerDF[ticker], 'LogPx', .395, 50*i, 500 )
    
    
    

# plot_output = pd.DataFrame()
# min_dim = .1
# max_dim = .9
# min_windows = 7
# max_windows = 8
# increments=20

# for fraction in range(0,increments):
#     for cutoffWindow in range (min_windows,max_windows):
#         plot_output.loc[str(min_dim+(fraction/increments)*(max_dim-min_dim)),str(100*cutoffWindow)] = fracDiffTest((min_dim+(fraction/increments)*(max_dim-min_dim)), cutoffWindow*100)







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
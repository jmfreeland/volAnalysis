from scipy.optimize import minimize
#attempt to optimize lookback, there appeared to be an improvement around 1100



def fracDiffTest(fraction, cutoffWindowDaily):
    ticker='SPY'
    #fracDiff[ticker] = {}
    temp_data = pd.Series()
    weights = fractionalWeights(cutoffWindowDaily,fraction)
    tickerDF[ticker].loc[:,'LogPx'] = np.log(tickerDF[ticker].loc[:,'Close'])
    tickerDF[ticker].loc[:,'LogPx'].iloc[-1*cutoffWindowDaily:].dot(weights)
    for i in range (cutoffWindowDaily,tickerDF[ticker].shape[0]):
        temp_data[tickerDF[ticker].index[i]] = (tickerDF[ticker].loc[:,'LogPx'].iloc[-1*cutoffWindowDaily+i:i]).dot(weights)
        #print(-1*cutoffWindowDaily+i)
    #print(i)
    result = adfuller(temp_data, autolag='AIC')
    return result[1]


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

dimension_list = {}
for ticker in test_stocks:
    dimension_list[ticker] = fracDiffOpt(tickerDF[ticker], 'logPx', 800, .8, 50, .01, 200)
    
#test window performance
ticker='SPY'
window_perf = pd.Series()
for i in range (1,50):
    [tmp, window_perf.loc[str(i)]] = fracDiff(tickerDF[ticker], 'LogPx', .395, 50*i, 500 )
    
    
    

plot_output = pd.DataFrame()
min_dim = .1
max_dim = .9
min_windows = 7
max_windows = 8
increments=20

for fraction in range(0,increments):
    for cutoffWindow in range (min_windows,max_windows):
        plot_output.loc[str(min_dim+(fraction/increments)*(max_dim-min_dim)),str(100*cutoffWindow)] = fracDiffTest((min_dim+(fraction/increments)*(max_dim-min_dim)), cutoffWindow*100)


fracDiffOpt(tickerDF['TLT'], 'LogPx', 750, .99, 25, .01, 500 )



# temp_data = temp_data - temp_data.mean()
# integrated_data = pd.Series()
# weights = fractionalWeights(cutoffWindowDaily,-1*fraction)
# for i in range (cutoffWindowDaily, temp_data.shape[0]):
#     integrated_data[tickerDF[ticker].index[i]] = (temp_data.iloc[i-cutoffWindowDaily:i]).dot(weights)
#     #print(-1*cutoffWindowDaily+i)
#     print(i)
result = adfuller(temp_data)
print(result[1])
plt.plot(temp_data) 
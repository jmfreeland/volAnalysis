# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:32:56 2020

@author: freel
"""

import scipy.stats as st

#helper function from https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
def get_best_distribution(data):
    dist_names = ['norm', 'exponweib', 'weibull_max', 'weibull_min', 'pareto', 'genextreme', 'gamma', 'beta', 'rayleigh']
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    #print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]
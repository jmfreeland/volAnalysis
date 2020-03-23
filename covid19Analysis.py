# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:17:02 2020

@author: freel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

#import data from hopkins set
confirmed_data = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
confirmed_global = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')


confirmed_cases_global = confirmed_global.sum()
confirmed_cases_global = confirmed_cases_global[5:]
days =  range(len(confirmed_global.columns[5:])-1)

ydata = np.array(confirmed_cases_global, dtype=float)
ydata = ydata / 7794798739
xdata = np.array(days, dtype=float)
xdata_long = np.array(range(len(confirmed_global.columns[5:])*2), dtype=float)


def sir_model(y, x, beta, gamma):
    S = -beta * y[0] * y[1] / N
    R = gamma * y[1]
    I = -(S + R)
    return S, I, R

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]

N = 1.0
I0 = ydata[0]
S0 = N - I0
R0 = 1

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata_long, *popt)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata_long, fitted)
plt.show()
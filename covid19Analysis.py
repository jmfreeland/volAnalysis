# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:17:02 2020
todo: automate population data

@author: freel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize


def sir_model(y, x, beta, gamma):
    S = -beta * y[0] * y[1] / N
    R = gamma * y[1]
    I = -(S + R)
    return S, I, R

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]


#import data from hopkins set
confirmed = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
confirmed_global = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
recovered = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
deaths = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

#fit a global model


confirmed_cases_global = confirmed.sum()
confirmed_cases_global = confirmed_cases_global[5:]
recovered_cases_global = recovered.sum()
recovered_cases_global = recovered_cases_global[5:]
deaths_global = deaths.sum()
deaths_global = deaths_global[5:]
active_cases_global = confirmed_cases_global - deaths_global - recovered_cases_global

days =  range(len(active_cases_global.index))

ydata = np.array(active_cases_global, dtype=float)
ydata = ydata / 7794798739
xdata = np.array(days, dtype=float)
xdata_long = np.array(range(len(active_cases_global.index)+20), dtype=float)

N = 1.0
I0 = ydata[0]
S0 = N - I0
R0 = 2

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata_long, *popt)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata_long, fitted)
plt.show()

#fit individual country models
model_countries = ['Italy','US','United Kingdom','China']
population = {}
population['Italy'] = 60480000000
population['US'] = 327300000000
population['United Kingdom'] = 66400000000
population['China'] = 1386000000

country_fit = {}

for country in model_countries:
    country_cases = confirmed_data.loc[confirmed_global['Country/Region']==country].sum()
    country_cases = country_cases[5:]
    #print(country_cases)
    days =  range(len(country_cases.index))
    
    ydata = np.array(country_cases, dtype=float)
    ydata = ydata / population[country]
    xdata = np.array(days, dtype=float)
    xdata_long = np.array(range(len(country_cases.index)*2), dtype=float)

    N = 1.0
    I0 = ydata[0]
    S0 = N - I0
    R0 = 2
    
    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    fitted = fit_odeint(xdata_long, *popt)

    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata_long, fitted)
    plt.show()
        
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


# #ODE for timestep in SIR model
# def sir_model(y, x, beta, gamma):
#     S = -beta * y[1] * y[0] / N
#     R = gamma * y[1]
#     I = -(S + R)
#     return S, I, R

#ODE for timestep in SIR model
def sir_model(y, x, beta, gamma):
    #initial conditions
    Si = y[0]
    Ii = y[1]
    Ri = y[2]
    #deltas
    S = (-beta * Si * Ii) / N
    I = ((beta * Ii * Si) / N) - gamma * Ii
    R = gamma * Ii
    return S, I, R


#import data from hopkins set
confirmed = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
confirmed_global = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
recovered = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
deaths = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

#fit a global model

global_population = 7794798739

confirmed_cases_global = confirmed.sum()
confirmed_cases_global = confirmed_cases_global[5:]
recovered_cases_global = recovered.sum()
recovered_cases_global = recovered_cases_global[5:]
deaths_global = deaths.sum()
deaths_global = deaths_global[5:]
active_cases_global = confirmed_cases_global - deaths_global - recovered_cases_global

days =  range(len(active_cases_global.index))

ydata = np.array(active_cases_global, dtype=float)
ydata = ydata / global_population
xdata = np.array(days, dtype=float)
xdata_long = np.array(range(len(active_cases_global.index)+20), dtype=float)

N = 1.0
I0 = ydata[0]
S0 = N - I0
R0 = recovered_cases_global[0]/global_population

#add initial conditions to definition
def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata, p0=[0.0,0.0])

fitted = fit_odeint(xdata_long, *popt)

plt.yscale=('linear')
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
country_active_cases = {}
for country in model_countries:
    country_cases = confirmed.loc[confirmed['Country/Region']==country].sum()
    country_recovered = recovered.loc[recovered['Country/Region']==country].sum()
    country_deaths = deaths.loc[deaths['Country/Region']==country].sum()
    country_active = country_cases[5:] - country_recovered[5:] - country_deaths[5:]
    country_active_cases[country] = country_active
    days =  range(len(country_active.index))
    
    ydata = np.array(country_active, dtype=float)
    ydata = ydata / population[country]
    xdata = np.array(days, dtype=float)
    xdata_long = np.array(range(len(country_active.index)+20), dtype=float)

    N = 1.0
    I0 = ydata[0]
    S0 = N - I0
    R0 = 3
    
    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    fitted = fit_odeint(xdata_long, *popt)

    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata_long, fitted)
    #wplt.yscale('log')
    plt.show()
        
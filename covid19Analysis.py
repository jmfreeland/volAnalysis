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


#ODE for timestep in SIR model
def sir_model(y, x, beta, gamma):
    #initial conditions
    Si = y[0]
    Ii = y[1]
    Ri = y[2]
    #deltas
    S = -beta * Si * Ii
    I = (beta * Ii * Si) - gamma * Ii
    R = gamma * Ii
    return S, I, R


#fit a globla model, so far doesn't do wel
#import data from hopkins set
confirmed = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
#confirmed_global = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
recovered = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
deaths = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
#fit a global model

global_population = 7794798739



confirmed_cases_global = confirmed.sum()
confirmed_cases_global = confirmed_cases_global[5:]
recovered_cases_global = recovered.sum()
recovered_cases_global = recovered_cases_global[5:]
if(len(recovered_cases_global)<len(confirmed_cases_global)):
    last_rec_day = len(recovered_cases_global)
    recovered_cases_global.loc[confirmed_cases_global.index[-1]] = recovered_cases_global.iloc[last_rec_day-1]*recovered_cases_global.iloc[last_rec_day-1]/recovered_cases_global.iloc[last_rec_day-2]
deaths_global = deaths.sum()
deaths_global = deaths_global[5:]
active_cases_global = pd.Series(confirmed_cases_global.values - deaths_global.values - recovered_cases_global.values, index = confirmed_cases_global.index)


days =  range(len(active_cases_global.index))

ydata = np.array(active_cases_global, dtype=float)
ydata = ydata / global_population
xdata = np.array(days, dtype=float)
xdata_long = np.array(range(200), dtype=float)

N = 1.0
I0 = ydata[0]
R0 = recovered_cases_global[0]/global_population
S0 = N - I0 - R0

#add initial conditions to definition. return integrated ODE for Infected cases
def fit_odeint(initial, x, beta, gamma):
    return integrate.odeint(sir_model, initial, x, args=(beta, gamma))[:,1]

#define function to optimize using global S0, I0, R0
def temp_odeint(x, beta, gamma):
    return fit_odeint((S0, I0, R0), x, beta, gamma)

#fit the curve using actual data
popt, pcov = optimize.curve_fit(temp_odeint, xdata, ydata, p0=[3,3])
print("Global N: " + str(N), ' S0' + str(S0) + ' I0: ' + str(I0) + 'R0: ' + str(R0))
print(xdata)
print(ydata) 
print(popt)

fitted = temp_odeint(xdata_long, *popt)

plt.yscale('log')
plt.plot(xdata, ydata, 'o')
plt.plot(xdata_long, fitted)
plt.show()

#fit individual country models
#model_countries = ['Italy','Spain','United Kingdom', 'China', 'US']
model_countries = ['Italy', 'US', 'Spain', 'China', 'United Kingdom', 'Korea, South', 'Germany','Iran', 'Brazil','France','India','Indonesia','Pakistan']
population = {}
population['Italy'] = 60480000
population['Spain'] = 46600000
population['US'] = 327300000
population['United Kingdom'] = 66400000
population['China'] = 1386000000
population['Korea, South'] = 51470000
population['Germany'] = 82790000
population['Iran'] = 81160000
population['Brazil'] = 209300000
population['France'] = 66900000
population['India'] = 1339000000
population['Indonesia'] = 273000000
population['Pakistan'] = 220000000

start_day = {}
start_day['Italy'] = 31
start_day['US'] = 33
start_day['Spain'] = 32
start_day['China'] = 6
start_day['United Kingdom'] = 34
start_day['Korea, South'] = 26
start_day['Germany'] = 33
start_day['Iran'] = 30
start_day['Brazil'] = 30
start_day['France'] = 33
start_day['India'] = 30
start_day['Indonesia'] = 30
start_day['Pakistan'] = 30

country_fit = {}
country_active_cases = {}
country_S0 = {}
country_R0 = {}
country_I0 = {}


#find proper start dates for each
for country in model_countries:
    country_cases = confirmed.loc[confirmed['Country/Region']==country].sum()
    country_recovered = recovered.loc[recovered['Country/Region']==country].sum()
    if(len(country_recovered)<len(country_cases)):
        last_rec_day = len(country_recovered)
        country_recovered.loc[country_cases.index[-1]] = country_recovered.iloc[last_rec_day-1]*country_recovered.iloc[last_rec_day-1]/country_recovered.iloc[last_rec_day-2]
    
    country_deaths = deaths.loc[deaths['Country/Region']==country].sum()
    country_active = pd.Series(1 + country_cases[5:].values - country_recovered[5:].values - country_deaths[5:].values, index=country_cases.index[5:])
    country_active_cases[country] = country_active
    country_R0[country] = (country_recovered[4+start_day[country]] + country_deaths[4+start_day[country]]) / population[country]
    country_I0[country] = country_active[4+start_day[country]] / population[country] 
    country_S0[country] = 1 - country_R0[country] - country_I0[country]
    
    days =  range(len(country_active.index))
    
    ydata = np.array(country_active)
    ydata = ydata / population[country]
    xdata = np.array(days)
    xdata_long = np.array(range(len(country_active.index)+20))

    xdata = xdata[start_day[country]:]
    ydata = ydata[start_day[country]:]
    xdata_long = xdata_long[start_day[country]:]
    

    
    #define function to optimize using country level S0, I0, R0
    def temp_odeint(x, beta, gamma):
        return fit_odeint((country_S0[country], country_I0[country], country_R0[country]), x, beta, gamma)

    popt, pcov = optimize.curve_fit(temp_odeint, xdata, ydata, bounds=(0,np.inf), p0=[3,0])
    fitted = temp_odeint(xdata_long, *popt)
    
    peak = np.argmax(fitted)
    print(country + ' Peak' + str(peak+start_day[country]))

    print(country + ' S0:' + str(country_S0[country]) + ' I0: ' + str(country_I0[country]) + ' R0: ' + str(country_R0[country]))
    print(xdata)
    #print(ydata)
    print(popt) 

    plt.figure()
    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata_long, fitted)
    plt.title(country)
    plt.yscale('log')
        
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:25:06 2022

@author: hp
"""

# import modules for performing file read and visualiztion
import numpy as npy
import pandas as pds
import matplotlib.pyplot as pylt
from sklearn.linear_model import LinearRegression


# reads data from csv file
rain_data = pds.read_csv('inputdata3.csv', delimiter=',')
print("\nInput File Data: \n", rain_data)

# splits and stores data into numpy array
a_rain = npy.array(rain_data['Rainfall'])
b_prod = npy.array(rain_data['Productivity'])

# reshapes the array
a_rain = a_rain.reshape(-1,1)
b_prod = b_prod.reshape(-1,1)


# fits data with y=b0+b1*x line
n_obs = len(a_rain)
a_mean = a_rain.mean()
b_mean = b_prod.mean()
slope_num = ((a_rain - a_mean) * (b_prod - b_mean)).sum()
slope_dens = ((a_rain - a_mean)**2).sum()
b_slope = slope_num / slope_dens
a_intercept = b_mean - (b_slope * a_mean)

#Creates window for plotting
pylt.figure(figsize=(10,8))

#Adds scatter plot y(x)
pylt.scatter(a_rain, b_prod, s=50, c=b_prod, linewidths=4, marker='D', label='Sample Data')
pylt.colorbar()

linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(a_rain,b_prod)
print("\nIntercept of Regression Line: ", linear_model.intercept_)
print("\nSlope of Regression Line: ", linear_model.coef_)
print("The Linear Model is: B = {:.5} + {:.5}A".format(linear_model.intercept_[0], linear_model.coef_[0][0]))

a_nvalue = 350
b_nvalue = ((linear_model.coef_[0]) * a_nvalue) + linear_model.intercept_
print("\nValue of y: \n", b_nvalue)


#Creates B array corresponding to the fitting line
b_fitting = a_intercept + (b_slope * a_rain)

#Plots the line fitting graph
pylt.plot(a_rain, b_fitting, linewidth=4.8, label='Regression Line')
pylt.scatter(a_mean, b_mean, c='r', linewidth=3.8, label='Average Point')
pylt.scatter(a_nvalue, b_nvalue, c='r', linewidth=3.8, label=f'b_nvalue = {((linear_model.coef_[0]) * 350) + (linear_model.intercept_)}')

#gives title and label to the graph
pylt.title('Rainfall v/s Productivity Coefficient', fontsize=13)
pylt.xlabel('Rainfall(mm)', fontsize=13)
pylt.ylabel('Productivity Coefficient', fontsize=13)
pylt.legend()
pylt.show()



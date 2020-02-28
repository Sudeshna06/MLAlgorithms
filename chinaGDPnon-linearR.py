# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:17:12 2020

@author: Sudeshna Bhakat
"""
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#regression model
def sigmoidFunc(x,beta1,beta2):
    y=1/(1+np.exp(-beta1*(x-beta2)))
    return y
df=pd.read_csv("china_gdp.csv")
df.fillna(0,inplace=True)
x=df['Year'].values
y=df['Value'].values
# normalizing data to bring data in the same range
xdata =x/max(x)
ydata =y/max(y)
'''curve_fit function uses non-linear least squares to fit sigmoid funtion to data, 
popt - list of optimal parameters
pcov - covariance of popt'''
popt, pcov = curve_fit(sigmoidFunc, xdata, ydata)
#sigmoid curve on the whole data
y = sigmoidFunc(xdata, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(xdata,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#breaking the dataset into train and test data
xTrain, xTest, yTrain,yTest = train_test_split(xdata,ydata,test_size=0.25)
#calculating optimal parameters
popt,pcov=curve_fit(sigmoidFunc,xTrain,yTrain)
#predicting values based on test data and regression model
yPred=sigmoidFunc(xTest,*popt)
#calculating accuracy
score=r2_score(yPred,yTest)
print("score: ",float(score))
#sorting the data for line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(xTest,yPred), key=sort_axis)
xTest, yPred = zip(*sorted_zip)
plt.plot(xTest, yTest, 'ro', label='data')
plt.plot(xTest,yPred, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
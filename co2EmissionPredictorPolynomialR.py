# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:01:38 2020

@author: Sudeshna Bhakat
"""

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
df = pd.read_csv("FuelConsumption.csv")
print(df.isnull().sum())
df.fillna(0,inplace=True)
print(df.isnull().any())
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
x=np.asanyarray(cdf[['ENGINESIZE']])
y=np.asanyarray(cdf[['CO2EMISSIONS']])
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.20)
#generate polynomial features of degree 2
poly=PolynomialFeatures(degree=2)
xTrainPoly=poly.fit_transform(xTrain)
regr=linear_model.LinearRegression()
regr.fit(xTrainPoly,yTrain)
print(regr.coef_)
print(regr.intercept_)
xTestPoly=poly.fit_transform(xTest)
yPredict=regr.predict(xTestPoly)
mse=mean_squared_error(yTest,yPredict)
print("mean squared error: ",float(mse))
score=r2_score(yTest,yPredict)
print("r2_score: ",score)
#plot the predicted line on the data
plt.scatter(x,y,color="green")
#sort x points for line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(xTest,yPredict), key=sort_axis)
xTest, yPredict = zip(*sorted_zip)
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.plot(xTest, yPredict, color='red')
plt.show()
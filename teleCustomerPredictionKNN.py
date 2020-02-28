# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:30:04 2020

@author: Sudeshna Bhakat
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv("teleCust1000t.csv")
print(df.head(10))
print(df['custcat'].value_counts())
#df.hist(column='income',bins=30)
print(df.columns)
#to use scikit learn convert pandas dataframe to numpy array. this array contains all value in float
x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed','employ', 'retire', 'gender', 'reside']].values
print(x[0:5])
ydata=df['custcat'].values
'''Data Standardization give data zero mean and unit variance, it is good practice,
especially for algorithms such as KNN which is based on distance of cases'''
scaler=preprocessing.StandardScaler().fit(x)
xdata=scaler.transform(x.astype(float))
print(xdata[0:5])
xTrain, xTest, yTrain, yTest = train_test_split(xdata,ydata,test_size=0.25,random_state=4)
print ('Train set:', xTrain.shape,  yTrain.shape)
print ('Test set:', xTest.shape,  yTest.shape)
k=9
neigh=KNeighborsClassifier(n_neighbors=k).fit(xTrain,yTrain)
yPredict=neigh.predict(xTest)
accuracyTrain=metrics.accuracy_score(yTrain,neigh.predict(xTrain))
accuracyTest=metrics.accuracy_score(yTest,yPredict)
print("Train data accuracy :%.2f",accuracyTrain)
print("Test data accuracy: %.2f",accuracyTest)
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(xTrain,yTrain)
    yPredict=neigh.predict(xTest)
    mean_acc[n-1] = metrics.accuracy_score(yTest, yPredict)

    
    std_acc[n-1]=np.std(yPredict==yTest)/np.sqrt(yPredict.shape[0])

print(mean_acc)
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
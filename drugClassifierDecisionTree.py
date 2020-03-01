# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:43 2020

@author: Sudeshna Bhakat
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data=pd.read_csv("drug200.csv")
#print(data.shape)
print(data.columns)
xdata=data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
ydata=data[['Drug']].values
'''some features in this dataset are categorical such as Sex, BP and cholestrol. 
Sklearn Decision Trees do not handle categorical variables.'''
le_sex=preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
xdata[:,1]=le_sex.transform(xdata[:,1])
le_bp=preprocessing.LabelEncoder()
le_bp.fit(['LOW','NORMAL','HIGH'])
xdata[:,2]=le_bp.transform(xdata[:,2])
le_chol=preprocessing.LabelEncoder()
le_chol.fit(['HIGH','NORMAL'])
xdata[:,3]=le_chol.transform(xdata[:,3])
print(xdata[0:5])
xTrain,xTest,yTrain,yTest=train_test_split(xdata,ydata,test_size=0.25,random_state=3)
drugTree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
#drugTree=DecisionTreeClassifier(criterion="gini",max_depth=4)
drugTree.fit(xTrain,yTrain)
yPred=drugTree.predict(xTest)
accuracy=metrics.accuracy_score(yTest,yPred)
#tocalculate accuracy without using sklearn
#accuracy=np.mean(yPred==yTest)
print(accuracy)
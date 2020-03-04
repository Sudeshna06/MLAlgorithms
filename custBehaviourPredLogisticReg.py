# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:34:45 2020

@author: Sudeshna Bhakat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split
churn_df=pd.read_csv("ChurnData.csv")
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
#convert target variable to integer type as sklearn processes integer values
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())
xdata=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].values
ydata=churn_df[['churn']].values
scaler=preprocessing.StandardScaler()
trans=scaler.fit(xdata)
xdata=trans.transform(xdata)
print(xdata[0:5])
xTrain,xTest,yTrain,yTest=train_test_split(xdata,ydata,test_size=0.2,random_state=4)
lr=LogisticRegression(C=0.01, solver='liblinear').fit(xTrain,yTrain)
yPred=lr.predict(xTest)
yPredProb=lr.predict_proba(xTest)
print(yPredProb)
print(jaccard_similarity_score(yTest,yPred))
cnf_matrix = confusion_matrix(yTest,yPred,labels=[1,0])
print(cnf_matrix)
print(classification_report(yTest,yPred))
print(log_loss(yTest,yPredProb))
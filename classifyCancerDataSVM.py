# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:54:54 2020

@author: Sudeshna Bhakat
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv("cell_samples.csv")
ax=df[df['Class']==4][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Red',label='malignant')
df[df['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Yellow',label='benign',ax=ax)
plt.show()
print(df.size)
'''certain values in 'BareNuc' column are '?'. converting them to nan values and then dropping them'''
df=df[pd.to_numeric(df['BareNuc'],errors='coerce').notnull()]
'''converting the 'BareNuc' attribute to int64 datatype'''
df['BareNuc']=df['BareNuc'].astype('int64')
print(df.dtypes)
print(df.size)
feature=df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
'''coverting feature set to array'''
xdata=np.asarray(feature)
'''converting class variable to array'''
#df=df['Class'].astype('int64')
ydata=np.asarray(df['Class'])
xTrain,xTest,yTrain,yTest=train_test_split(xdata,ydata,test_size=0.25,random_state=4)
classify=svm.SVC(kernel='linear', gamma='auto')
'''classify=svm.SVC(kernel='rbf', gamma='auto')'''
classify.fit(xTrain,yTrain)
yPred=classify.predict(xTest)
'''generating confusion matrix'''
cnfMat=metrics.confusion_matrix(yTest,yPred,labels=[2,4])
print(cnfMat)
'''generating classifying report'''
classReport=metrics.classification_report(yTest,yPred)
print(classReport)
'''generating only f1 score'''
print(metrics.f1_score(yTest,yPred,average='weighted'))
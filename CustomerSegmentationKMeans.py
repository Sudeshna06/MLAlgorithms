# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:14:17 2020

@author: Sudeshna Bhakat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('Cust_Segmentation.csv')
'''Address here is categorical variable and KMeans is not applicable to categorical so dropping that attribute'''
dfNew=df.drop('Address',axis=1)
'''normalizing the data so that the datapoint is distributed equally'''
x=dfNew.values[:,1:]
x=np.nan_to_num(x)
scaler=StandardScaler()
scaler.fit(x)
xdata=scaler.transform(x)
print(xdata)
'''k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. 
The customers in each cluster are similar to each other demographically. 
Now we can create a profile for each group, considering the common characteristics of each cluster. 
For example, the 3 clusters can be:
AFFLUENT, EDUCATED AND OLD AGED
MIDDLE AGED AND MIDDLE INCOME
YOUNG AND LOW INCOME'''
k=3
'''applying KMeans algorithm to the data'''
kmeans=KMeans(init='k-means++',n_clusters=k,n_init=12)
kmeans.fit(xdata)
kmeansLabel=kmeans.labels_
kmeansCenter=kmeans.cluster_centers_
dfNew["labels"]=kmeansLabel
print(dfNew)
print(dfNew.groupby('labels').mean())
'''distribution of customers based on age and income'''
area = np.pi * ( xdata[:, 1])**2  
plt.scatter(xdata[:, 0], xdata[:, 3], s=area, c=kmeansLabel.astype(np.float), alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import warnings
warnings.filterwarnings(action='ignore') 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sb
import math
import kmedoids

# purity function
def purity(clusters, classes):
    cm = np.array(pd.crosstab(clusters, classes))
    return np.sum(np.amax(cm, axis=1)) / np.sum(cm)

def NanData_cleaning(data):
    print(data.isnull().sum())
    for row in range(len(data)):
        for column in range(len(data.columns)):
            if isinstance(data.iloc[row, column], str) is False:
                if math.isnan(data.iloc[row, column]):
                    if column == 3 or column == 6 or column == 7 or column == 8 or column == 9:  # numerical
                        data.iat[row, column] = round(data.iloc[:, column].mean())
                    else:  # categorical
                        data.iat[row, column] = data.iloc[:, column].mode()[0]

    return data


def Outlier_cleaning(data,inlist):
    
    for indi in inlist:
        sb.distplot(data[indi])
        plt.show()
        value=data[indi]
        stDev=np.std(value)
        m=np.mean(value)
        outlier_index=data[data[indi]>m+3*stDev][indi].index
        data=data.drop(outlier_index,axis=0)
        sb.distplot(data[indi])
        plt.show()

    return data

def Data_normalization(data,inlist):
    minMax = MinMaxScaler()
    
    for indi in inlist:
        data[indi]=minMax.fit_transform(np.array(data[indi]).reshape(-1,1)).reshape(-1)

    return data

def make_train_set(dataframe,indicators,country):
    year=dataframe.columns[4:]
    name=[]
    for c in country:
        for y in year:
            name.append(c+y)
    data=pd.DataFrame({'country':name})
    
    for indi in range(len(indicators)):
        values=[]
        for i in range(len(dataframe)):
            if(dataframe['Indicator Name'].at[i]==indicators[indi]):
                for y in year:
                    values.append(dataframe[y].at[i])
        data[indicators[indi]]=values
      
    data=NanData_cleaning(data)
    
    data=Outlier_cleaning(data,indicators)
    
    data=Data_normalization(data,indicators)

    return data

def get_data_indicators(dataframe, year, indicators,country):
    data=pd.DataFrame({'country':c})
    
    for indi in range(len(indicators)):
        values=[]
        for i in range(len(dataframe)):
            if(dataframe['Indicator Name'].at[i]==indicators[indi]):
                values.append(dataframe[str(year)].at[i])
        data[indicators[indi]]=values
        values=[]
        
    return data
def labeling(data):
    values=[]

    
    for i in range(len(data)):
        if(data[i]<7):
            values.append(0)
        elif(7<=data[i]<14):
            values.append(1)
        elif(14<=data[i]<20):
            values.append(2)
        elif(data[i]>=20):
            values.append(3)

    return values

#load dataset
wdi = pd.read_csv("WDIData.csv")

#list of country names
c=wdi['Country Code'].unique()

#set year
year=2013

#indicators list
list=['Population ages 65 and above (% of total population)',
      'Survival to age 65, female (% of cohort)','Survival to age 65, male (% of cohort)',
      'Population growth (annual %)','GNI (constant 2010 US$)','Life expectancy at birth, total (years)',
      'Birth rate, crude (per 1,000 people)']
train=make_train_set(wdi,list,c)
train=train.drop(['Population ages 65 and above (% of total population)'],axis=1)
train=train.drop(['country'],axis=1)
#select indicators
data=get_data_indicators(wdi,year,list,c)



#preprocessing
# clean nandata
data = NanData_cleaning(data)
data['target']=labeling(data['Population ages 65 and above (% of total population)'])


#store country
c=data['country'].unique()

#data reduction
data = data.drop(['country'], axis=1)

# drop outlier
data = Outlier_cleaning(data,data.columns)

target=data['target']
data=data.drop(['target'],axis=1)

# data normalization
data = Data_normalization(data,data.columns)
data=data.drop(['Population ages 65 and above (% of total population)'],axis=1)

# kmeans model
print('K-means clustering')
print()
nc=[4]
mi=[300, 200, 100]
for n in nc:
    for i in mi:
        print('n_clusters:',n)
        print('max_iter:',i)
        # create model and prediction
        kmeans = KMeans(n_clusters=n,algorithm='auto',max_iter=i)
        kmeans.fit(train)
        predict = pd.DataFrame(kmeans.predict(data))
        predict.columns=['predict']
        #print(predict)
        print("purity:",purity(predict['predict'], target))

        # concatenate labels to df as a new column
        r = pd.concat([data,predict],axis=1)
        # plot the cluster assignments
        plt.scatter(r['Life expectancy at birth, total (years)'],r['GNI (constant 2010 US$)'],c=r['predict'],cmap="plasma")
        plt.show()
        print()
#kmedoids model
distances = pairwise_distances(data, metric='euclidean')

M, C = kmedoids.kMedoids(distances, 4)
predict = np.zeros(len(data))
for label in C:
    for point_idx in C[label]:
        predict[point_idx] = label

predict = pd.DataFrame(predict)
predict.columns=['predict']

print("purity:",purity(predict['predict'], target))

plt.scatter(data['Life expectancy at birth, total (years)'], data['GNI (constant 2010 US$)'], c=predict['predict'], cmap="plasma")
plt.show()
print()

# DBSCAN
print('DBSCAN')
print()
ep=[0.02,0.05, 0.1,0.5]
ms=[10,15, 20,30,50,100]
for e in ep:
    for m in ms:
        print('eps:',e)
        print('min_samples:',m)
        # create model and prediction
        dbscan = DBSCAN(eps=e,min_samples=m,p=4)
        predict = dbscan.fit_predict(data)
        print("purity:",purity(predict, target))
        # plot the cluster assignments
        plt.scatter(data['Life expectancy at birth, total (years)'], data['GNI (constant 2010 US$)'], c=predict, cmap="plasma")
        plt.show()
        print()


# In[ ]:





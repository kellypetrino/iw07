#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np


# In[6]:


# Read in metadata
meta = pd.read_csv("metadata.csv")
meta["DATE"] = pd.to_datetime(meta["DATE"])
meta = meta.drop_duplicates()
meta.index = range(len(meta))
meta


# In[58]:


# Convert seasons to ints in metadata
for i in range(len(meta20)):
    v = meta20.iloc[i]['SEASON']
    if v == 'CHRISTMAS':
        v = 1
    elif v == 'CHRISTMAS PEAK':
        v = 2
    elif v == 'COLUMBUS DAY':
        v = 3
    elif v == 'EASTER':
        v = 4
    elif v == 'FALL':
        v = 5
    elif v == 'HALLOWEEN':
        v = 6
    elif v == 'JERSEY WEEK':
        v = 7
    elif v == 'JULY 4th':
        v = 8
    elif v == 'MARDI GRAS':
        v = 9
    elif v == 'MARTIN LUTHER KING JUNIOR DAY':
        v = 10
    elif v == 'MEMORIAL DAY':
        v = 11
    elif v == 'PRESIDENTS WEEK':
        v = 12
    elif v == 'SEPTEMBER LOW':
        v = 13
    elif v == 'SPRING':
        v = 14
    elif v == 'SUMMER BREAK':
        v = 15
    elif v == 'THANKSGIVING':
        v = 16
    elif v == 'WINTER':
        v = 17
    else:
        v = 18
    meta20.iat[i,7] = v
    
meta20['SEASON'] = meta20['SEASON'].astype(int)


# In[8]:


# Restrict to only necesary metadata
meta_season = meta[['DAYOFWEEK','SEASON','MKHOURSEMH','EPHOURSEMH','HSHOURSEMH','AKHOURSEMH','WDWMINTEMP_mean','WEATHER_WDWPRECIP']]
# Split data to train and test sets
train_season, test_season = train_test_split(meta_season)
# K-means cluster on train set
kmeans_season = KMeans(n_clusters=35, random_state=0).fit(train_season)


# In[9]:


meta


# In[10]:


# Number of days in each grouping
counts = []
for i in range(35):
    counts.append(len(kmeans_season.labels_[kmeans_season.labels_==i]))
print(counts)


# In[11]:


count = pd.DataFrame(counts, columns = ["COUNTS"])


# In[12]:


train_season["CLUSTER"] = kmeans_season.labels_


# In[13]:


train_season[train_season["CLUSTER"] == 5]


# In[14]:


test_preds = kmeans_season.predict(test_season)


# In[15]:


# Number of days in each grouping
test_counts = []
for i in range(35):
    test_counts.append(len(test_preds[test_preds==i]))
print(test_counts)


# In[16]:


test_count = pd.DataFrame(test_counts, columns = ["COUNTS"])
test_count


# In[17]:


test_season["PREDS"] = test_preds


# In[18]:


test_season["DATE"] = meta.iloc[test_season.index,0]
train_season["DATE"] = meta.iloc[train_season.index,0]
train_season = train_season.sort_values(by="DATE")
test_season = test_season.sort_values(by="DATE")


# In[19]:


train_season.to_csv('clean/train.csv', index=False)
test_season.to_csv('clean/test.csv', index=False)


# In[20]:


train_season[890:920]


# In[ ]:


train_season[train_season["CLUSTER"]==24]


# In[21]:


test_season


# In[57]:


# Read in 2020 metadata
meta20 = pd.read_csv("2020/metadata.csv")
meta20["DATE"] = pd.to_datetime(meta20["DATE"])
meta20 = meta20.drop_duplicates()
meta20.index = range(len(meta20))
# meta20 = meta20[(meta20["DATE"] >= pd.Timestamp(2019, 10, 31)) |((meta20["DATE"] >= pd.Timestamp(2019, 7, 13)) & (meta20["DATE"] <= pd.Timestamp(2019, 7, 20)))]
meta20 = meta20[(meta20["DATE"] >= pd.Timestamp(2019, 1, 1)) & (meta20["DATE"] <= pd.Timestamp(2019, 3, 13))]
meta20


# In[59]:


# Restrict to only necesary metadata
meta20_test = meta20[['DAYOFWEEK','SEASON','MKHOURSEMH','EPHOURSEMH','HSHOURSEMH','AKHOURSEMH','WDWMINTEMP_mean','WEATHER_WDWPRECIP']]
test20_preds = kmeans_season.predict(meta20_test)
test20_preds


# In[60]:


meta20_test["PREDS"] = test20_preds
meta20_test


# In[61]:


meta20_test["DATE"] = meta.iloc[meta20_test.index,0]
meta20_test = meta20_test.sort_values(by="DATE")
meta20_test


# In[62]:


for i in range(len(meta20_test)):
    meta20_test.iat[i,9] = meta20_test.iloc[i]["DATE"].replace(year=2020)
    
meta20_test


# In[63]:


meta20_test.to_csv('clean/test20pre.csv', index=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[140]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import math
import time


# In[41]:


# Defining paths to ride data 
RIDES = {'MK': {'DWARFS': '7_dwarfs_train.csv', 'PIRATES': 'pirates_of_caribbean.csv', 'SPLASH': 'splash_mountain.csv'},
         'EP': {'SOARIN': 'soarin.csv', 'SPACE': 'spaceship_earth.csv'},
         'HS': {'ALIEN': 'alien_saucers.csv', 'ROCKN': 'rock_n_rollercoaster.csv', 'SLINKY': 'slinky_dog.csv', 'TSM': 'toy_story_mania.csv'},
         'AK': {'FLIGHT': 'flight_of_passage.csv', 'DINO': 'dinosaur.csv', 'EVEREST': 'expedition_everest.csv', 'SAFARI': 'kilimanjaro_safaris.csv', 'NAVI': 'navi_river.csv'}
        }


# In[49]:


# Defining "new" rides for exponential decay (opened in last 10 years)
NEW_RIDES = {'DWARFS': pd.Timestamp(2014, 5, 28), 
             'ALIEN': pd.Timestamp(2018, 6, 30), 'SLINKY': pd.Timestamp(2018, 6, 30), 
             'FLIGHT': pd.Timestamp(2017, 5, 27), 'NAVI': pd.Timestamp(2017, 5, 27)
            }


# In[317]:


# Load in full datasets
mk = pd.read_csv('clean/mk.csv')
ep = pd.read_csv('clean/ep.csv')
hs = pd.read_csv('clean/hs.csv')
ak = pd.read_csv('clean/ak.csv')

mk["DATE"] = pd.to_datetime(mk["DATE"])
ep["DATE"] = pd.to_datetime(ep["DATE"])
hs["DATE"] = pd.to_datetime(hs["DATE"])
ak["DATE"] = pd.to_datetime(ak["DATE"])
mk["DATETIME"] = pd.to_datetime(mk["DATETIME"])
ep["DATETIME"] = pd.to_datetime(ep["DATETIME"])
hs["DATETIME"] = pd.to_datetime(hs["DATETIME"])
ak["DATETIME"] = pd.to_datetime(ak["DATETIME"])


# In[60]:


ak.head()


# In[68]:


# Load in clusters
train = pd.read_csv('clean/train.csv')
test = pd.read_csv('clean/test.csv')


# In[414]:


# Function to retroactively predict best path on a given day 
def computePath(park, date, data):
    paths = []
    start_time = []
    total_time = []
    full = data[data["DATE"]==date]
#     print(full)
    names = list(RIDES[park].keys())
    for ride in names:
        if ride in NEW_RIDES.keys():
            if NEW_RIDES[ride] > date:
                names.remove(ride)
                full = full.drop([ride], axis=1)
#     print(names)
    # Starting from each time
    for i in range(len(full)-10):
        temp = full[names].iloc[i]
        # Starting from each ride
        for k, val in enumerate(temp):
    #         print(temp)
            if math.isnan(val):
                continue
            current_time = full['DATETIME'].iloc[i] + timedelta(minutes=val)
            rides = names[k]
            time = val
            not_added = names.copy()
            not_added.remove(names[k])
            temp = full[full['DATETIME'] >= current_time].iloc[0][not_added].astype(float)
            for j in range(5):
                if math.isnan(temp.min()):
                    break
                r = temp.idxmin()
                t = temp.min()
                if current_time + timedelta(minutes=t) > full['DATETIME'].iloc[-1]:
                    break
                current_time = current_time + timedelta(minutes=t)
                rides = rides + r + ' '
                time = time + t
                not_added.remove(r)
                temp = full[full['DATETIME'] >= current_time].iloc[0][not_added].astype(float)
                while math.isnan(temp.min()) and j != 4:
                    if current_time + timedelta(seconds=60) >= full['DATETIME'].iloc[-1]:
                        break
                    current_time = current_time + timedelta(seconds=60)
                    temp = full[full['DATETIME'] >= current_time].iloc[0][not_added].astype(float)
            if len(not_added) > 0:
                continue
            start_time.append(full['DATETIME'].iloc[i])
            paths.append(rides)
            total_time.append(time)
    idx = total_time.index(min(total_time))
    print(paths[idx])
    print(start_time[idx].time())
    print(total_time[idx])
    
    return paths[idx], start_time[idx], total_time[idx]


# In[415]:


# TESTING
computePath('HS', pd.Timestamp(2015, 12, 15), hs)
computePath('MK', pd.Timestamp(2016, 12, 8), mk)


# In[78]:


mk.info()


# In[129]:


# Take averages across days in a dataset
def getClusterWaits(park_data, dates):
    data = park_data[park_data["DATE"].isin(dates)]
    data = data.drop(["DATE"], axis=1)
#     print(data)
    for i in range(len(data)):
        if data.iloc[i]["DATETIME"].hour >= 5:
            data.iat[i,0] = data.iloc[i]["DATETIME"].replace(year=2020, month=1, day=1, second=0)
        else:
            data.iat[i,0] = data.iloc[i]["DATETIME"].replace(year=2020, month=1, day=2, second=0)
    # Group by times and take averages
#     print(data)
    avg = data.groupby("DATETIME").mean()
    print(avg)
    return avg


# In[131]:


#### Create array of datasets for training clusters 
mk_clusters = []
ep_clusters = []
hs_clusters = []
ak_clusters = []
start = time.time()
for i in range(35):
    print("CLUSTER", i)
    dates = train[train["CLUSTER"] == i]["DATE"]
    mk_clusters.append(getClusterWaits(mk, dates))
    ep_clusters.append(getClusterWaits(ep, dates))
    hs_clusters.append(getClusterWaits(hs, dates))
    ak_clusters.append(getClusterWaits(ak, dates))
    
end = time.time()
print("TIME TO COMPUTE CLUSTERS", end-start)
    


# In[208]:


# Fix cluster structure
for i, cluster in enumerate(ak_clusters):
    print(cluster)
    ak_clusters[i] = cluster.reset_index(level=["DATETIME"])


# In[ ]:


# Function to retroactively predict best path on a given day 
def predictPath(park, date, data):
    paths = []
    start_time = []
    total_time = []
    full = data.copy()
#     print(full)
    names = list(RIDES[park].keys())
    for ride in names:
        if ride in NEW_RIDES.keys():
            if NEW_RIDES[ride] > date:
                names.remove(ride)
                full = full.drop([ride], axis=1)
#     print(names)
    # Starting from each time
    for i in range(len(full)-10):
        temp = full[names].iloc[i]
        # Starting from each ride
        for k, val in enumerate(temp):
    #         print(temp)
            if math.isnan(val):
                continue
            current_time = full['DATETIME'].iloc[i] + timedelta(minutes=val)
            rides = names[k]
            time = val
            not_added = names.copy()
            not_added.remove(names[k])
            temp = full[full['DATETIME'] >= current_time].iloc[0][not_added].astype(float)
            for j in range(5):
                if math.isnan(temp.min()):
                    break
                r = temp.idxmin()
                t = temp.min()
                if current_time + timedelta(minutes=t) > full['DATETIME'].iloc[-1]:
                    break
                current_time = current_time + timedelta(minutes=t)
                rides = rides + r + ' '
                time = time + t
                not_added.remove(r)
                temp = full[full['DATETIME'] >= current_time].iloc[0][not_added].astype(float)
                while math.isnan(temp.min()) and j != 4:
                    if current_time + timedelta(seconds=60) >= full['DATETIME'].iloc[-1]:
                        break
                    current_time = current_time + timedelta(seconds=60)
                    temp = full[full['DATETIME'] >= current_time].iloc[0][not_added].astype(float)
            if len(not_added) > 0:
                continue
            start_time.append(full['DATETIME'].iloc[i])
            paths.append(rides)
            total_time.append(time)

    idx = total_time.index(min(total_time))
    print(paths[idx])
    print(start_time[idx].time())
    print(total_time[idx])
    
    return paths[idx], start_time[idx], total_time[idx]


# In[349]:


## Predict a path for one day
def predictMK(date, clusters, test, mk):
    c = test[test["DATE"] == date]["PREDS"].iloc[0]
    data = mk_clusters[c]
    opening = mk[mk["DATE"] == date].iloc[0]["DATETIME"].replace(year=2020, month=1, day=1)
    closing = mk[mk["DATE"] == date].iloc[-1]["DATETIME"].replace(year=2020, month=1, day=1)
    if closing.hour < 5:
        closing = closing.replace(day=2)
    data = data[(data["DATETIME"] >= opening) & (data["DATETIME"] <= closing)]
    data["DATETIME"] = pd.to_datetime(data["DATETIME"])
#     print(data)
    return predictPath("MK", pd.to_datetime(date), data)

def predictEP(date, clusters, test, ep):
    c = test[test["DATE"] == date]["PREDS"].iloc[0]
    data = ep_clusters[c]
    opening = ep[ep["DATE"] == date].iloc[0]["DATETIME"].replace(year=2020, month=1, day=1)
    closing = ep[ep["DATE"] == date].iloc[-1]["DATETIME"].replace(year=2020, month=1, day=1)
    if closing.hour < 5:
        closing = closing.replace(day=2)
    data = data[(data["DATETIME"] >= opening) & (data["DATETIME"] <= closing)]
    data["DATETIME"] = pd.to_datetime(data["DATETIME"])
#     print(data)
    return predictPath("EP", pd.to_datetime(date), data)

def predictHS(date, clusters, test, hs):
    c = test[test["DATE"] == date]["PREDS"].iloc[0]
    data = hs_clusters[c]
    opening = hs[hs["DATE"] == date].iloc[0]["DATETIME"].replace(year=2020, month=1, day=1)
    closing = hs[hs["DATE"] == date].iloc[-1]["DATETIME"].replace(year=2020, month=1, day=1)
    if closing.hour < 5:
        closing = closing.replace(day=2)
    data = data[(data["DATETIME"] >= opening) & (data["DATETIME"] <= closing)]
    data["DATETIME"] = pd.to_datetime(data["DATETIME"])
#     print(data)
    return predictPath("HS", pd.to_datetime(date), data)
    
def predictAK(date, clusters, test, ak):
    c = test[test["DATE"] == date]["PREDS"].iloc[0]
    data = ak_clusters[c]
    opening = ak[ak["DATE"] == date].iloc[0]["DATETIME"].replace(year=2020, month=1, day=1)
    closing = ak[ak["DATE"] == date].iloc[-1]["DATETIME"].replace(year=2020, month=1, day=1)
    if closing.hour < 5:
        closing = closing.replace(day=2)
    data = data[(data["DATETIME"] >= opening) & (data["DATETIME"] <= closing)]
    data["DATETIME"] = pd.to_datetime(data["DATETIME"])
#     print(data)
    return predictPath("AK", pd.to_datetime(date), data)


# In[267]:


idx = np.random.randint(len(test))
date = test.iloc[idx]["DATE"]
print(date)
predictMK(date, mk_clusters)


# In[224]:


computePath('MK', pd.to_datetime(test.iloc[idx]["DATE"]), mk)


# In[214]:


mk_clusters[0].info()


# In[327]:


## Accuracy 
def isAccurate(pred_path, pred_start, actual_total, date, data, thresh):
    rides = pred_path.split()
    current_time = pred_start.replace(year=date.year, month=date.month, day=date.day)
    comp_time = 0
    for ride in rides:
        i = data.index[data['DATETIME'] >= current_time].tolist()
        t = data.iloc[i[0]][ride]
        j = 1
        while math.isnan(t):
            t = data.iloc[i[j]][ride]
            j += 1
        current_time = current_time + timedelta(minutes=t)
        comp_time = comp_time + t
    print("Comp time:", comp_time)
    return(abs(comp_time - actual_total) <= thresh)
    


# In[250]:


## TESTING
pred_path = "PIRATES SPLASH DWARFS "
pred_start = pd.Timestamp(2020, 1, 1, 8, 25)
actual_total = 25.0

isAccurate(pred_path, pred_start, actual_total, pd.Timestamp(2019, 8, 6), mk)


# In[299]:


# Script to iterate through subset of test in MK and compute accuracy
mk_count_acc30 = 0
mk_count_acc25 = 0
mk_count_acc20 = 0
mk_count_acc15 = 0
mk_count_acc45 = 0
mk_count_acc60 = 0
idx = np.random.randint(0,len(test),30)
sub = test.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('MK', pd.to_datetime(date), mk)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictMK(date, mk_clusters)
    mk_count_acc30 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk, 30)
    mk_count_acc25 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk, 25)
    mk_count_acc20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk, 20)
    mk_count_acc15 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk, 15)
    mk_count_acc45 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk, 45)
    mk_count_acc60 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk, 60)


# In[300]:


mk_acc30.append(mk_count_acc30 / 30)
mk_acc25.append(mk_count_acc25 / 30)
mk_acc20.append(mk_count_acc20 / 30)
mk_acc15.append(mk_count_acc15 / 30)
mk_acc45.append(mk_count_acc45 / 30)
mk_acc60.append(mk_count_acc60 / 30)

print(mk_acc30)
print(mk_acc25)
print(mk_acc20)
print(mk_acc15)
print(mk_acc45)
print(mk_acc60)


# In[302]:


# Script to iterate through subset of test in EP and compute accuracy
ep_count_acc30 = 0
ep_count_acc25 = 0
ep_count_acc20 = 0
ep_count_acc15 = 0
ep_count_acc45 = 0
ep_count_acc60 = 0
idx = np.random.randint(0,len(test),30)
sub = test.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('EP', pd.to_datetime(date), ep)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictEP(date, ep_clusters)
    ep_count_acc30 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep, 30)
    ep_count_acc25 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep, 25)
    ep_count_acc20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep, 20)
    ep_count_acc15 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep, 15)
    ep_count_acc45 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep, 45)
    ep_count_acc60 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep, 60)


# In[303]:


ep_acc30.append(ep_count_acc30 / 30)
ep_acc25.append(ep_count_acc25 / 30)
ep_acc20.append(ep_count_acc20 / 30)
ep_acc15.append(ep_count_acc15 / 30)
ep_acc45.append(ep_count_acc45 / 30)
ep_acc60.append(ep_count_acc60 / 30)

print(ep_acc30)
print(ep_acc25)
print(ep_acc20)
print(ep_acc15)
print(ep_acc45)
print(ep_acc60)


# In[304]:


# Script to iterate through subset of test in EP and compute accuracy
hs_count_acc30 = 0
hs_count_acc25 = 0
hs_count_acc20 = 0
hs_count_acc15 = 0
hs_count_acc45 = 0
hs_count_acc60 = 0
idx = np.random.randint(0,len(test),30)
sub = test.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('HS', pd.to_datetime(date), hs)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictHS(date, hs_clusters)
    hs_count_acc30 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs, 30)
    hs_count_acc25 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs, 25)
    hs_count_acc20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs, 20)
    hs_count_acc15 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs, 15)
    hs_count_acc45 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs, 45)
    hs_count_acc60 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs, 60)


# In[305]:


hs_acc30.append(hs_count_acc30 / 30)
hs_acc25.append(hs_count_acc25 / 30)
hs_acc20.append(hs_count_acc20 / 30)
hs_acc15.append(hs_count_acc15 / 30)
hs_acc45.append(hs_count_acc45 / 30)
hs_acc60.append(hs_count_acc60 / 30)

print(hs_acc30)
print(hs_acc25)
print(hs_acc20)
print(hs_acc15)
print(hs_acc45)
print(hs_acc60)


# In[306]:


# Script to iterate through subset of test in EP and compute accuracy
ak_count_acc30 = 0
ak_count_acc25 = 0
ak_count_acc20 = 0
ak_count_acc15 = 0
ak_count_acc45 = 0
ak_count_acc60 = 0
idx = np.random.randint(0,len(test),30)
sub = test.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('AK', pd.to_datetime(date), ak)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictAK(date, ak_clusters)
    ak_count_acc30 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak, 30)
    ak_count_acc25 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak, 25)
    ak_count_acc20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak, 20)
    ak_count_acc15 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak, 15)
    ak_count_acc45 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak, 45)
    ak_count_acc60 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak, 60)


# In[307]:


ak_acc30.append(ak_count_acc30 / 30)
ak_acc25.append(ak_count_acc25 / 30)
ak_acc20.append(ak_count_acc20 / 30)
ak_acc15.append(ak_count_acc15 / 30)
ak_acc45.append(ak_count_acc45 / 30)
ak_acc60.append(ak_count_acc60 / 30)

print(ak_acc30)
print(ak_acc25)
print(ak_acc20)
print(ak_acc15)
print(ak_acc45)
print(ak_acc60)


# In[401]:


## 2020!!!
# Load in full datasets
mk20 = pd.read_csv('clean/mk20.csv')
ep20 = pd.read_csv('clean/ep20.csv')
hs20 = pd.read_csv('clean/hs20.csv')
ak20 = pd.read_csv('clean/ak20.csv')

mk20["DATE"] = pd.to_datetime(mk20["DATE"])
ep20["DATE"] = pd.to_datetime(ep20["DATE"])
hs20["DATE"] = pd.to_datetime(hs20["DATE"])
ak20["DATE"] = pd.to_datetime(ak20["DATE"])
mk20["DATETIME"] = pd.to_datetime(mk20["DATETIME"])
ep20["DATETIME"] = pd.to_datetime(ep20["DATETIME"])
hs20["DATETIME"] = pd.to_datetime(hs20["DATETIME"])
ak20["DATETIME"] = pd.to_datetime(ak20["DATETIME"])

# Load in clusters
test20 = pd.read_csv('clean/test20.csv')


# In[404]:


# Script to iterate through subset of test in MK and compute accuracy
mk_count_acc30_20 = 0
mk_count_acc25_20 = 0
mk_count_acc20_20 = 0
mk_count_acc15_20 = 0
mk_count_acc45_20 = 0
mk_count_acc60_20 = 0
idx = np.random.randint(0,len(test20),20)
sub = test20.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('MK', pd.to_datetime(date), mk20)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictMK(date, mk_clusters, test20, mk20)
    mk_count_acc30_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk20, 30)
    mk_count_acc25_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk20, 25)
    mk_count_acc20_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk20, 20)
    mk_count_acc15_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk20, 15)
    mk_count_acc45_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk20, 45)
    mk_count_acc60_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), mk20, 60)


# In[ ]:


mk_acc30_20=[]
mk_acc25_20=[]
mk_acc20_20=[]
mk_acc15_20=[]
mk_acc45_20=[]
mk_acc60_20=[]

mk_acc30_20.append(mk_count_acc30_20 / 20)
mk_acc25_20.append(mk_count_acc25_20 / 20)
mk_acc20_20.append(mk_count_acc20_20 / 20)
mk_acc15_20.append(mk_count_acc15_20 / 20)
mk_acc45_20.append(mk_count_acc45_20 / 20)
mk_acc60_20.append(mk_count_acc60_20 / 20)

print(mk_acc30_20)
print(mk_acc25_20)
print(mk_acc20_20)
print(mk_acc15_20)
print(mk_acc45_20)
print(mk_acc60_20)


# In[ ]:


# Script to iterate through subset of test in MK and compute accuracy
ep_count_acc30_20 = 0
ep_count_acc25_20 = 0
ep_count_acc20_20 = 0
ep_count_acc15_20 = 0
ep_count_acc45_20 = 0
ep_count_acc60_20 = 0
idx = np.random.randint(0,len(test20),20)
sub = test20.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('EP', pd.to_datetime(date), ep20)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictEP(date, mk_clusters, test20, ep20)
    ep_count_acc30_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep20, 30)
    ep_count_acc25_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep20, 25)
    ep_count_acc20_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep20, 20)
    ep_count_acc15_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep20, 15)
    ep_count_acc45_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep20, 45)
    ep_count_acc60_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ep20, 60)


# In[ ]:


ep_acc30_20=[]
ep_acc25_20=[]
ep_acc20_20=[]
ep_acc15_20=[]
ep_acc45_20=[]
ep_acc60_20=[]

ep_acc30_20.append(ep_count_acc30_20 / 20)
ep_acc25_20.append(ep_count_acc25_20 / 20)
ep_acc20_20.append(ep_count_acc20_20 / 20)
ep_acc15_20.append(ep_count_acc15_20 / 20)
ep_acc45_20.append(ep_count_acc45_20 / 20)
ep_acc60_20.append(ep_count_acc60_20 / 20)

print(ep_acc30_20)
print(ep_acc25_20)
print(ep_acc20_20)
print(ep_acc15_20)
print(ep_acc45_20)
print(ep_acc60_20)


# In[ ]:


# Script to iterate through subset of test in MK and compute accuracy
hs_count_acc30_20 = 0
hs_count_acc25_20 = 0
hs_count_acc20_20 = 0
hs_count_acc15_20 = 0
hs_count_acc45_20 = 0
hs_count_acc60_20 = 0
idx = np.random.randint(0,len(test20),20)
sub = test20.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('HS', pd.to_datetime(date), hs20)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictHS(date, mk_clusters, test20, hs20)
    hs_count_acc30_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs20, 30)
    hs_count_acc25_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs20, 25)
    hs_count_acc20_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs20, 20)
    hs_count_acc15_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs20, 15)
    hs_count_acc45_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs20, 45)
    hs_count_acc60_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), hs20, 60)


# In[ ]:


hs_acc30_20=[]
hs_acc25_20=[]
hs_acc20_20=[]
hs_acc15_20=[]
hs_acc45_20=[]
hs_acc60_20=[]

hs_acc30_20.append(hs_count_acc30_20 / 20)
hs_acc25_20.append(hs_count_acc25_20 / 20)
hs_acc20_20.append(hs_count_acc20_20 / 20)
hs_acc15_20.append(hs_count_acc15_20 / 20)
hs_acc45_20.append(hs_count_acc45_20 / 20)
hs_acc60_20.append(hs_count_acc60_20 / 20)

print(hs_acc30_20)
print(hs_acc25_20)
print(hs_acc20_20)
print(hs_acc15_20)
print(hs_acc45_20)
print(hs_acc60_20)


# In[ ]:


# Script to iterate through subset of test in MK and compute accuracy
ak_count_acc30_20 = 0
ak_count_acc25_20 = 0
ak_count_acc20_20 = 0
ak_count_acc15_20 = 0
ak_count_acc45_20 = 0
ak_count_acc60_20 = 0
idx = np.random.randint(0,len(test20),20)
sub = test20.iloc[idx,:]
for date in sub["DATE"]:
    # Compute actual pred
    print("#####", date, "#####")
    print("ACTUAL")
    act_path, act_start, act_total = computePath('AK', pd.to_datetime(date), ak20)
    print("PREDICT")
    pred_path, pred_start, pred_total = predictAK(date, mk_clusters, test20, ak20)
    ak_count_acc30_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak20, 30)
    ak_count_acc25_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak20, 25)
    ak_count_acc20_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak20, 20)
    ak_count_acc15_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak20, 15)
    ak_count_acc45_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak20, 45)
    ak_count_acc60_20 += isAccurate(pred_path, pred_start, act_total, pd.to_datetime(date), ak20, 60)


# In[ ]:


ak_acc30_20=[]
ak_acc25_20=[]
ak_acc20_20=[]
ak_acc15_20=[]
ak_acc45_20=[]
ak_acc60_20=[]

ak_acc30_20.append(ak_count_acc30_20 / 20)
ak_acc25_20.append(ak_count_acc25_20 / 20)
ak_acc20_20.append(ak_count_acc20_20 / 20)
ak_acc15_20.append(ak_count_acc15_20 / 20)
ak_acc45_20.append(ak_count_acc45_20 / 20)
ak_acc60_20.append(ak_count_acc60_20 / 20)

print(ak_acc30_20)
print(ak_acc25_20)
print(ak_acc20_20)
print(ak_acc15_20)
print(ak_acc45_20)
print(ak_acc60_20)


# In[ ]:





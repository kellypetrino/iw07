#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import math


# In[3]:


# Defining paths to ride data 
RIDES = {'MK': {'DWARFS': '7_dwarfs_train.csv', 'PIRATES': 'pirates_of_caribbean.csv', 'SPLASH': 'splash_mountain.csv'},
         'EP': {'SOARIN': 'soarin.csv', 'SPACE': 'spaceship_earth.csv'},
         'HS': {'ALIEN': 'alien_saucers.csv', 'ROCKN': 'rock_n_rollercoaster.csv', 'SLINKY': 'slinky_dog.csv', 'TSM': 'toy_story_mania.csv'},
         'AK': {'FLIGHT': 'flight_of_passage.csv', 'DINO': 'dinosaur.csv', 'EVEREST': 'expedition_everest.csv', 'SAFARI': 'kilimanjaro_safaris.csv', 'NAVI': 'navi_river.csv'}
        }


# In[102]:


# Defining "new" rides for exponential decay (opened in last 10 years)
NEW_RIDES = {'DWARFS': pd.Timestamp(2014, 5, 28), 
             'ALIEN': pd.Timestamp(2018, 6, 30), 'SLINKY': pd.Timestamp(2018, 6, 30), 
             'FLIGHT': pd.Timestamp(2017, 5, 27), 'NAVI': pd.Timestamp(2017, 5, 27)
            }


# In[34]:


# Helper functions
def dropDownTimes(ride):
    new = ride[ride["SPOSTMIN"] != -999]
    new.dropna(subset=["SPOSTMIN"], inplace=True)
    return new
def dropActTimes(ride):
    return ride.drop(['SACTMIN'],axis=1)
def sortTimesByDate(ride, date):
    ride_date = ride[ride["DATE"] == date]
    ride_date = ride_date.drop(['SACTMIN','DATE'],axis=1)
    return ride_date.sort_values(by=["SPOSTMIN"])


# In[54]:


# Cleaning data to remove down times and unnecessary data
def clean(ride):
    ride.columns = ["DATE", "DATETIME", "SACTMIN", "SPOSTMIN"]
    ride["DATE"] = pd.to_datetime(ride["DATE"])
    ride["DATETIME"] = pd.to_datetime(ride["DATETIME"])
    new = dropActTimes(ride)
    new = dropDownTimes(new)
#     print(new)
    return new 


# In[142]:


# Get value from day before
def getDayBefore(datetime, ride, data):
    newdatetime = datetime - pd.offsets.DateOffset(days=1)
    subdata = data[data["DATETIME"] >= newdatetime]
    if len(subdata) > 0:
        return subdata.iloc[0][ride]
    else:
        return math.nan
# Get value from day after
def getDayAfter(datetime, ride, data):
    newdatetime = datetime + pd.offsets.DateOffset(days=1)
    subdata = data[data["DATETIME"] >= newdatetime]
    if len(subdata) > 0:
        return subdata.iloc[0][ride]
    else:
        return math.nan
# Get value from year before
def getYearBefore(datetime, ride, data):
    newdatetime = datetime - pd.offsets.DateOffset(years=1)
    subdata = data[data["DATETIME"] >= newdatetime]
    if len(subdata) > 0:
        return subdata.iloc[0][ride]
    else:
        return math.nan
# Get value from year after
def getYearAfter(datetime, ride, data):
    newdatetime = datetime + pd.offsets.DateOffset(years=1)
    subdata = data[data["DATETIME"] >= newdatetime]
    if len(subdata) > 0:
        return subdata.iloc[0][ride]
    else:
        return math.nan
    
# Fill in missing values
def fill(park, data):
    startdate = data.iloc[0]["DATE"]
    lastdate = data.iloc[len(data)-1]["DATE"]
    startyear = startdate.year
    lastyear = lastdate.year
    for i in range(len(data)):
        date = data.iloc[i]["DATE"]
        year = date.year
        datetime = data.iloc[i]["DATETIME"]
        print(datetime)
        for j, ride in enumerate(data.columns[2:]):
            if ride in NEW_RIDES.keys():
                if NEW_RIDES[ride] > date:
                    continue 
            if math.isnan(data.iloc[i][ride]):
                daybefore = math.nan
                dayafter = math.nan
                yearbefore = math.nan
                yearafter = math.nan
                # Fill with average of time before and after
                if i > 0 and data.iloc[i-1]["DATE"] == date and not math.isnan(data.iloc[i-1][ride]) and i < len(data) - 1 and data.iloc[i+1]["DATE"] == date and not math.isnan(data.iloc[i+1][ride]):
                    # Take average
                    data.iat[i, j+2] = (data.iloc[i-1][ride] + data.iloc[i+1][ride]) / 2
                    continue
#                     print(data.iloc[i][ride])                        
                # Fill with average of day before and after
                if date > startdate and date < lastdate:
                    daybefore = getDayBefore(datetime, ride, data)
                    dayafter = getDayAfter(datetime, ride, data)
                    if not math.isnan(daybefore) and not math.isnan(dayafter):
                        data.iat[i, j+2] = (daybefore + dayafter) / 2
                        continue
                # Fill with values from previous year 
                if year > startyear:
                    yearbefore = getYearBefore(datetime, ride, data)
                    if not math.isnan(yearbefore):
                        # Exponential decay
                        if ride in NEW_RIDES.keys():
                            t = year - NEW_RIDES[ride].year
                            data.iat[i, j+2] = yearbefore - 60*math.exp(-t)
                        else:
                            data.iat[i, j+2] = yearbefore
                        continue
                ## Fill with values from next year
                if year < lastyear:
                    yearafter = getYearAfter(datetime, ride, data)
                    if not math.isnan(yearafter):
                        # Exponential growth
                        if ride in NEW_RIDES.keys() and NEW_RIDES[ride].year <= year:
                            t = year - NEW_RIDES[ride].year
                            data.iat[i, j+2] = yearafter + 60*math.exp(-t)
                        elif ride in NEW_RIDES.keys() and NEW_RIDES[ride].year > year:
                            data.iat[i, j+2] = math.nan
                        else:
                            data.iat[i, j+2] = yearafter
                        continue

               
                # EDGE CASES
                ## First time of day empty 
                if i > 0 and i < len(data) - 1 and data.iloc[i-1]["DATE"] != date and not math.isnan(data.iloc[i+1][ride]):
                    data.iat[i, j+2] = data.iloc[i+1][ride]
                    continue
                ## Last time of day empty
                if i > 0 and i < len(data) - 1 and data.iloc[i+1]["DATE"] != date and not math.isnan(data.iloc[i-1][ride]):
                    data.iat[i, j+2] = data.iloc[i-1][ride]
                    continue
                ## No day before 
                if date == startdate and not math.isnan(dayafter):
                    data.iat[i, j+2] = dayafter
                    continue
                ## No day after 
                if date == lastdate and not math.isnan(daybefore):
                    data.iat[i, j+2] = daybefore
                    continue
    return data
    


# In[121]:


import time
start = time.time()
mk = getWaitData('MK')
end = time.time()
print("TIME TO CLEAN MK:", end-start)
print(mk)
print(mk[mk["DATE"].isin([pd.Timestamp(2015, 6, 1)])])


# In[80]:


# Read in a parks ride info and clean it 
def getWaitData(park):
    full = pd.DataFrame()
    for ride, file in RIDES[park].items():
#         print(file)
        data = pd.read_csv(file)
        data = clean(data)
        if (full.empty):
            full = data
        else:
            full = pd.merge(full, data, how="outer", on=["DATE","DATETIME"])
    labels = list(RIDES[park].keys())
    labels.insert(0, 'DATETIME')
    labels.insert(0, 'DATE')
    full.columns = labels
    full = full.sort_values(by='DATETIME')
    full.index = range(len(full))
    return fill(park, full)
#     return full
        


# In[130]:


start = time.time()
mk3 = fill('MK', mk2)
end = time.time()
print("TIME TO CLEAN MK:", end-start)
mk3


# In[123]:


mk.isna().sum()


# In[128]:


mk2.isna().sum()


# In[131]:


mk3.isna().sum()


# In[153]:


mk3.to_csv('mk.csv', index=False)


# In[161]:


mk4 = fill('MK', mk3)
mk4


# In[163]:


mk4.isna().sum()


# In[164]:


mk5 = fill('MK', mk4)
mk5


# In[165]:


mk5.isna().sum()


# In[166]:


mk5.to_csv('clean/mk.csv', index=False)


# In[134]:


start = time.time()
ep = getWaitData('EP')
end = time.time()
print("TIME TO CLEAN EP:", end-start)
print(ep)


# In[135]:


ep.isna().sum()


# In[136]:


ep2 = fill('EP', ep)
ep2


# In[137]:


ep2.isna().sum()


# In[79]:


mk_original = getWaitData('MK')
# print(mk)
print(mk[mk["DATE"].isin([pd.Timestamp(2015, 6, 1)])])
# ep = getWaitData('EP')
# print(ep)
# hs = getWaitData('HS')
# print(hs)
# ak = getWaitData('AK')
# print(ak)


# In[152]:


ep2.to_csv('clean/ep.csv', index=False)


# In[143]:


start = time.time()
hs = getWaitData('HS')
end = time.time()
print("TIME TO CLEAN HS:", end-start)
print(hs)


# In[145]:


hs.isna().sum()


# In[146]:


hs2 = fill('HS', hs)
hs2


# In[147]:


hs2.isna().sum()


# In[148]:


hs3 = fill('HS', hs2)
hs3


# In[149]:


hs3.isna().sum()


# In[151]:


hs3.to_csv('clean/hs.csv', index=False)


# In[154]:


start = time.time()
ak = getWaitData('AK')
end = time.time()
print("TIME TO CLEAN AK:", end-start)
print(ak)


# In[155]:


ak.isna().sum()


# In[156]:


ak2 = fill('AK', ak)
ak2


# In[157]:


ak2.isna().sum()


# In[158]:


ak3 = fill('AK', ak2)
ak3


# In[159]:


ak3.isna().sum()


# In[160]:


ak3.to_csv('clean/ak.csv', index=False)


# In[79]:


import math
def computePath(park, date):
    paths = []
    start_time = []
    total_time = []
    full = getWaitData(park, date)
#     print(full)
    names = list(RIDES[park].keys())
#     print(names)
    for i in range(len(full) - 20):
        temp = full[names].iloc[i]
#         print(temp)
        if math.isnan(temp.min()):
            continue
        current_time = full['DATETIME'].iloc[i]
        rides = ''
        time = 0
        not_added = names.copy()
        for j in range(5):
            if math.isnan(temp.min()):
                break
            r = temp.idxmin()
            t = temp.min()
            if current_time + timedelta(minutes=t) > full['DATETIME'].iloc[-1]:
                break
            current_time = current_time + timedelta(minutes=t)
            rides = rides + r + ' '
            time += t
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


# In[78]:


computePath('MK', pd.Timestamp(2015, 12, 15))


# In[84]:


computePath('MK', pd.Timestamp(2015, 12, 15))
computePath('MK', pd.Timestamp(2016, 12, 8))


# In[81]:


computePath('EP', pd.Timestamp(2015, 12, 15))
computePath('EP', pd.Timestamp(2016, 12, 8))


# In[82]:


computePath('HS', pd.Timestamp(2018, 12, 6))
computePath('HS', pd.Timestamp(2019, 12, 16))


# In[83]:


computePath('AK', pd.Timestamp(2018, 12, 6))
computePath('AK', pd.Timestamp(2019, 12, 16))


# In[85]:


computePath('MK', pd.Timestamp(2019, 12, 16))
computePath('EP', pd.Timestamp(2019, 12, 16))


# In[87]:


computePath('HS', pd.Timestamp(2018, 10, 30))
computePath('HS', pd.Timestamp(2019, 10, 30))
computePath('AK', pd.Timestamp(2018, 4, 3))
computePath('AK', pd.Timestamp(2019, 10, 30))


# In[86]:


computePath('MK', pd.Timestamp(2015, 10, 28))
computePath('MK', pd.Timestamp(2019, 10, 30))
computePath('EP', pd.Timestamp(2015, 10, 28))
computePath('EP', pd.Timestamp(2019, 10, 30))


# In[88]:


computePath('MK', pd.Timestamp(2018, 4, 3))
computePath('MK', pd.Timestamp(2019, 10, 30))


# In[89]:


computePath('AK', pd.Timestamp(2018, 10, 30))
computePath('AK', pd.Timestamp(2019, 10, 30))


# In[169]:


### 2020 DATA !!
import os
# Read in a parks ride info and clean it 
def getWaitData2020(park):
    full = pd.DataFrame()
    for ride, file in RIDES[park].items():
#         print(file)
        data = pd.read_csv(os.path.join('2020',file))
        data = clean(data)
        if (full.empty):
            full = data
        else:
            full = pd.merge(full, data, how="outer", on=["DATE","DATETIME"])
            
    full = full[full["DATE"] >= pd.Timestamp(2020, 1, 1)]
        
    labels = list(RIDES[park].keys())
    labels.insert(0, 'DATETIME')
    labels.insert(0, 'DATE')
    full.columns = labels
    full = full.sort_values(by='DATETIME')
    full.index = range(len(full))
    return fill(park, full)
#     return full
       


# In[170]:


mk20 = getWaitData2020('MK')
mk20


# In[175]:


mk20 = fill('MK', mk20)
mk20.isna().sum()


# In[176]:


mk20.to_csv('clean/mk20.csv')


# In[177]:


ep20 = getWaitData2020('EP')
ep20.isna().sum()


# In[180]:


ep20 = fill('EP', ep20)
ep20.isna().sum()


# In[181]:


ep20.to_csv('clean/ep20.csv')


# In[182]:


hs20 = getWaitData2020('HS')
hs20.isna().sum()


# In[185]:


hs20 = fill('HS', hs20)
hs20.isna().sum()


# In[186]:


hs20.to_csv('clean/hs20.csv')


# In[187]:


ak20 = getWaitData2020('AK')
ak20.isna().sum()


# In[190]:


ak20 = fill('AK', ak20)
ak20.isna().sum()


# In[191]:


ak20.to_csv('clean/ak20.csv')


# In[ ]:





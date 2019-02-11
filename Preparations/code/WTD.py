import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.basemap import Basemap
import sys
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = df_train[['PAX']]

def latlong_process(df_train):
    df_train['Distance'] = ""
    for index, row in df_train.iterrows():
        df_train.at[index, 'Distance'] = str(math.hypot(row['LatitudeArrival'] - row['LatitudeDeparture'], row['LongitudeArrival'] - row['LongitudeDeparture']))
        #start, finish = cardinal_direction_simple(row)
        start, finish = cardinal_direction_complex(row)
        #start, finish = cardinal_direction_specific_simple(row)
        df_train.at[index, 'Start'] = start
        df_train.at[index, 'Finish'] = finish

    return df_train

def cardinal_direction_simple(row):
    Start = ''
    Finish = ''

    if(row['LongitudeDeparture'] > 40):
        Start += "N"
    else:
        Start += "S"

    if(row['LatitudeDeparture'] > -100):
        Start += "E"
    else:
        Start += "W"

    if(row['LongitudeArrival'] > 40):
        Finish += "N"
    else:
        Finish += "S"

    if(row['LatitudeArrival'] > -100):
        Finish += "E"
    else:
        Finish += "W"

    return Start, Finish

def cardinal_direction_complex(row):
    Start = ''
    Finish = ''

    if(row['LongitudeDeparture'] > 45):
        Start += "N"
    elif(row['LongitudeDeparture'] < 35):
        Start += "S"
    else:
        Start += "C"

    if (row['LatitudeDeparture'] > -90):
        Start += "E"
    elif (row['LatitudeDeparture'] < -110):
        Start += "W"
    else:
        Start += "C"

    if (row['LongitudeArrival'] > 45):
        Finish += "N"
    elif (row['LongitudeArrival'] < 35):
        Finish += "S"
    else:
        Finish += "C"

    if (row['LongitudeArrival'] > -90):
        Finish += "E"
    elif (row['LongitudeArrival'] < -110):
        Finish += "W"
    else:
        Finish += "C"

    return Start, Finish

def cardinal_direction_specific_simple(row):
    Start = ''
    Finish = ''

    if (row['LongitudeDeparture'] > 45):
        Start += "N"
    elif (row['LongitudeDeparture'] < 35):
        Start += "S"

    if (row['LatitudeDeparture'] > -90):
        Start += "E"
    elif (row['LatitudeDeparture'] < -110):
        Start += "W"

    if Start == "":
        Start += 'C'

    if (row['LongitudeArrival'] > 45):
        Finish += "N"
    elif (row['LongitudeArrival'] < 35):
        Finish += "S"

    if (row['LongitudeArrival'] > -90):
        Finish += "E"
    elif (row['LongitudeArrival'] < -110):
        Finish += "W"

    if Finish == "":
        Finish += "C"

    return Start, Finish


def flight_direction(df_train):
    for index, row in df_train.iterrows():
        if row['Start'][0] == row['Finish'][0]:
            if row['Start'] == 'CC':
                df_train.at[index, 'Direction'] = 'E'
            else:
                df_train.at[index,'Direction'] = row['Finish'][1]
        elif row['Start'][1] == row['Finish'][1]:
            if row['Finish'][0] == 'C':
                if row['Start'][0] == 'N':
                    df_train.at[index, 'Direction'] = 'S'
                else:
                    df_train.at[index, 'Direction'] = 'N'
            else:
                df_train.at[index, 'Direction'] = row['Finish'][0]
        else:
            if row['Start'][0] == 'N':
                df_train.at[index, 'Direction'] = 'SE'
            elif row['Start'][0] == 'S':
                df_train.at[index, 'Direction'] = 'NE'
            elif row['Start'] == 'CC':
                df_train.at[index, 'Direction'] = row['Finish']
            elif row['Start'][0] == 'C':
                if row['Finish'][0] == 'N':
                    df_train.at[index, 'Direction'] = 'NE'
                elif row['Finish'][0] == 'S':
                    df_train.at[index, 'Direction'] = 'SE'
    return df_train


df_train = latlong_process(df_train)
df_train = flight_direction(df_train)
df_test = latlong_process(df_test)
df_test = flight_direction(df_test)
directions_start  = df_train.groupby('Departure')['Direction'].apply(lambda x: x.tolist())
plt.figure(figsize=(12, 5))
plt.hist(directions_start, bins=14)
plt.xlabel('Start')
plt.ylabel('Counter')
plt.grid(True)
plt.show()
directions_start  = df_test.groupby('Departure')['Direction'].apply(lambda x: x.tolist())
plt.figure(figsize=(12, 5))
plt.hist(directions_start, bins=14)
plt.xlabel('Start')
plt.ylabel('Counter')
plt.grid(True)
plt.show()
'''
directions_start  = df_train.groupby('PAX')['Start'].apply(lambda x: x.tolist())
plt.figure(figsize=(12, 5))
plt.hist(directions_start, bins=14)
plt.xlabel('Start')
plt.ylabel('Counter')
plt.grid(True)
plt.show()

directions_start  = df_test.groupby('Departure')['Start'].apply(lambda x: x.tolist())
plt.figure(figsize=(12, 5))
plt.hist(directions_start, bins=14)
plt.xlabel('Start')
plt.ylabel('Counter')
plt.grid(True)
plt.show()
directions_finish = df_test.groupby('Arrival')['Finish'].apply(lambda x: x.tolist())
plt.figure(figsize=(12, 5))
plt.hist(directions_finish, bins=7)
plt.xlabel('Finish')
plt.ylabel('Counter')
plt.grid(True)
plt.show()

# Map (long, lat) to (x, y) for plotting
locs = df_train[['CityDeparture', 'LongitudeDeparture', 'LatitudeDeparture']]
locs = locs.drop_duplicates()
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6,
            lat_0=45, lon_0=-100)
m.etopo(scale=0.5, alpha=0.5)
lon= locs['LongitudeDeparture'].values.tolist()
lat = locs['LatitudeDeparture'].values.tolist()

lon, lat = m(lat, lon)
m.scatter(lon, lat, marker = 'o', color='r', zorder=5)
'''
#plt.show()

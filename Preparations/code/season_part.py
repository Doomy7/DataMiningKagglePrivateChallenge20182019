import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from collections import Counter

df_train = pd.read_csv('train.csv')
y_train = df_train[['PAX']]
dates = df_train['DateOfDeparture']

dates = df_train['DateOfDeparture'].str.split('-', expand=True)
dates.columns = ['FYear', 'FMonth', 'FDay']
for index, row in dates.iterrows():
    dates.at[index, 'FD06'] = datetime.date(int(row['FYear']), int(row['FMonth']), int(row['FDay'])).weekday()
df_train = pd.concat([dates, df_train['PAX']], axis=1)

def semester():
    for index, row in df_train.iterrows():
        if(int(row['FMonth']) > 6):
            df_train.at[index, 'Semester'] = 2
        else:
            df_train.at[index, 'Semester'] = 1
    return df_train

def trimester():
    for index, row in df_train.iterrows():
        if(int(row['FMonth']) > 8):
            df_train.at[index, 'Trimester'] = 3
        elif(int(row['FMonth']) > 5):
            df_train.at[index, 'Trimester'] = 2
        else:
            df_train.at[index, 'Trimester'] = 1
    return df_train


def quarter():
    for index, row in df_train.iterrows():
        if(int(row['FMonth']) > 9):
            df_train.at[index, 'Quarter'] = 4
        elif(int(row['FMonth']) > 6):
            df_train.at[index, 'Quarter'] = 3
        elif(int(row['FMonth']) > 3):
            df_train.at[index, 'Quarter'] = 2
        else:
            df_train.at[index, 'Quarter'] = 1
    return df_train

def season():
    for index, row in df_train.iterrows():
        if(int(row['FMonth']) == 12 or 1 or 2):
            df_train.at[index, 'Season'] = 0
        elif(int(row['FMonth']) == 3 or 4 or 5):
            df_train.at[index, 'Season'] = 1
        elif(int(row['FMonth']) == 6 or 7 or 8):
            df_train.at[index, 'Season'] = 2
        else:
            df_train.at[index, 'Season'] = 3
    return df_train



def view(df_train):
    if 'Semester' in df_train:
        sem = df_train.groupby('PAX')['Semester'].apply(lambda x: x.tolist())
        plt.figure(figsize=(12, 5))
        plt.hist(sem, bins=4)
        plt.xlabel('Semester')
        plt.ylabel('Counter')
        plt.grid(True)
        plt.show()
    if 'Trimester' in df_train:
        tri = df_train.groupby('PAX')['Trimester'].apply(lambda x: x.tolist())
        plt.figure(figsize=(12, 5))
        plt.hist(tri, bins=4)
        plt.xlabel('Trimester')
        plt.ylabel('Counter')
        plt.grid(True)
        plt.show()
    if 'Quarter' in df_train:
        quar = df_train.groupby('PAX')['Quarter'].apply(lambda x: x.tolist())
        plt.figure(figsize=(12, 5))
        plt.hist(quar, bins=4)
        plt.xlabel('Quarter')
        plt.ylabel('Counter')
        plt.grid(True)
        plt.show()
    if 'Season' in df_train:
        sea = df_train.groupby('PAX')['Season'].apply(lambda x: x.tolist())
        plt.figure(figsize=(12, 5))
        plt.hist(sea, bins=4)
        plt.xlabel('Season')
        plt.ylabel('Counter')
        plt.grid(True)
        plt.show()

df_train = semester()
df_train = trimester()
df_train = quarter()
df_train = season()
view(df_train)
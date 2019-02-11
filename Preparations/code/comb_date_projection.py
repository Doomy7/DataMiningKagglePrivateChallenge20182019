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
dates = pd.concat([dates, df_train['PAX']], axis=1)

def build_years(y):
    year = dates.groupby('FYear')['PAX'].apply(lambda x: x.tolist())
    #year = dates.groupby('PAX')['FYear'].apply(lambda x: x.tolist())
    plt.figure(figsize=(12, 5))
    plt.xlabel("PAX")
    plt.ylabel("Counter")
    plt.grid(True)
    year.hist(bins=8)
    if y:
        plt.savefig(str(os.getcwd()) + '\\images\\Year' + str("YYYY") + ".png")
    else:
        plt.show()

def build_months(y):
    month = dates.groupby('FMonth')['PAX'].apply(lambda x: x.tolist())
    #month = dates.groupby('PAX')['FMonth'].apply(lambda x: x.tolist())
    plt.figure(figsize=(12, 5))
    plt.xlabel("PAX")
    plt.ylabel("Counter")
    plt.grid(True)
    month.hist(bins=8)
    if y:
        plt.savefig(str(os.getcwd()) + '\\images\\Month' + str("MM") + ".png")
    else:
        plt.show()

def build_days(y):
    day = dates.groupby('FDay')['PAX'].apply(lambda x: x.tolist())
    #day = dates.groupby('PAX')['FDay'].apply(lambda x: x.tolist())
    plt.figure(figsize=(12, 5))
    plt.xlabel("PAX")
    plt.ylabel("Counter")
    plt.grid(True)
    day.hist(bins=8)
    if y:
        plt.savefig(str(os.getcwd()) + '\\images\\Day' + str("DD") + ".png")
    else:
        plt.show()

def build_weekday(y):
    #weekday = dates.groupby('FD06')['PAX'].apply(lambda x: x.tolist())
    weekday = dates.groupby('PAX')['FD06'].apply(lambda x: x.tolist())
    plt.figure(figsize=(12, 5))
    plt.xlabel("WeekDay")
    plt.ylabel("Counter")
    plt.grid(True)
    weekday.hist(bins=7)
    if y:
        plt.savefig(str(os.getcwd()) + '\\images\\WeekDay' + str("WD") + ".png")
    else:
        plt.show()

build_years(0)
build_months(0)
build_days(0)
build_weekday(0)

for index, row in dates.iterrows():
    if int(row['FDay']) > 15:
        dates.at[index, 'BiWeekly'] = int(row['FMonth'])*2
    else:
        dates.at[index, 'BiWeekly'] = int(row['FMonth'])*2-1


biweek = dates.groupby('BiWeekly')['PAX'].apply(lambda x: x.tolist())
print(biweek)
plt.figure(figsize=(12, 5))
plt.xlabel("PAX")
plt.ylabel("Counter")
plt.grid(True)
biweek.hist(bins=8)
plt.show()
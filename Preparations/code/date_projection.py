import pandas as pd
import datetime
import matplotlib.pyplot as plt
from collections import Counter
import os
df_train = pd.read_csv('train.csv')
y_train = df_train[['PAX']]
dates = df_train['DateOfDeparture']

days = []
months = []
years = []
daycount = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

def dates_split():
    for date in dates:
        day = date.split('-')
        #days.append(datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday())
        days.append(int(day[2]))
        months.append(int(day[1]))
        years.append(int(day[0]))

def build_days():
    for index, row in df_train.iterrows():
        day = row['DateOfDeparture'].split('-')

        if (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 0):
            daycount[0].append(row['PAX'])
        elif (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 1):
            daycount[1].append(row['PAX'])
        elif (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 2):
            daycount[2].append(row['PAX'])
        elif (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 3):
            daycount[3].append(row['PAX'])
        elif (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 4):
            daycount[4].append(row['PAX'])
        elif (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 5):
            daycount[5].append(row['PAX'])
        elif (datetime.date(int(day[0]), int(day[1]), int(day[2])).weekday() == 6):
            daycount[6].append(row['PAX'])

def dayCounter(x):
    for day in daycount.keys():
        dayCount = Counter(daycount.get(day))
        plt.bar(dayCount.keys(), dayCount.values())
        plt.title("Day " + str(day))
        plt.xlabel("Flight category")
        plt.ylabel("Counter")
        plt.grid(True)
        if x:
          plt.savefig(str(os.getcwd()) + '\\images\\Day ' + str(day)+".png")
        else:
            plt.show()

def view_Years(x):
    plt.hist(years, bins = 5)
    plt.xlabel('Year')
    plt.ylabel('Counter')
    plt.grid(True)
    if x:
        plt.savefig(str(os.getcwd()) + "\\images\\Years.png")
    else:
        plt.show()

def view_Months(x):
    plt.hist(months, bins =23)
    plt.xlabel('Month')
    plt.ylabel('Counter')
    plt.grid(True)
    if x:
        plt.savefig(str(os.getcwd()) + "\\images\\Months.png")
    else:
        plt.show()

def view_Days(x):
    plt.figure(figsize=(12, 5))
    plt.hist(days, bins=61)
    plt.xlabel('Days')
    plt.ylabel('Counter')
    plt.grid(True)
    if x:
        plt.savefig(str(os.getcwd()) + "\\images\\Days.png")
    else:
        plt.show()

dates_split()
#view_Years(1)
view_Months(1)
#view_Days(1)
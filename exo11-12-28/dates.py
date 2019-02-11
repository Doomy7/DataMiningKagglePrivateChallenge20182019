import pandas as pd
import datetime
import matplotlib.pyplot as plt
from collections import Counter
df_train = pd.read_csv('train.csv')
intdays = []
intmonths = []
intyears = []
y_train = df_train[['PAX']]
dates = df_train[['DateOfDeparture']]

for date in dates['DateOfDeparture']:
    da = date.split('-')
    intdays.append(datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday())
    intmonths.append(int(da[1]))
    intyears.append(int(da[0]))

print(sum(intdays) / len(intdays))
# df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)
day0 = []
day1 = []
day2 = []
day3 = []
day4 = []
day5 = []
day6 = []

for index, row in df_train.iterrows():
    da = row['DateOfDeparture'].split('-')
    if (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 0):
        day0.append(row['PAX'])
    elif (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 1):
        day1.append(row['PAX'])
    elif (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 2):
        day2.append(row['PAX'])
    elif (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 3):
        day3.append(row['PAX'])
    elif (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 4):
        day4.append(row['PAX'])
    elif (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 5):
        day5.append(row['PAX'])
    elif (datetime.date(int(da[0]), int(da[1]), int(da[2])).weekday() == 6):
        day6.append(row['PAX'])

for i in range(0, 7):
    print(intdays.count(i))
print("")
for i in range(1, 13):
    print(intmonths.count(i))


daycount = [day0, day1, day2, day3, day4, day5, day6]

def dayC():
    for day in daycount:
        dayCounter = Counter(day)
        plt.bar(dayCounter.keys(), dayCounter.values())
        plt.show()


#plt.hist(daycount, bins=14)
# plt.show()
# plt.hist(intdays, bins=14)
# plt.show()
# plt.hist(intmonths, bins=24)
# plt.show()
yearC = Counter(intyears)
plt.bar(yearC.keys(), yearC.values())
plt.show()
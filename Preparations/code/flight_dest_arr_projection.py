import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from collections import Counter

df_train = pd.read_csv('train.csv')
y_train = df_train[['PAX']]
flights_dep = df_train[['Departure', 'PAX']]
flights_arr = df_train[['Arrival', 'PAX']]

f_dep = flights_dep.groupby('Departure')['PAX'].apply(lambda x: x.tolist())
f_arr = flights_arr.groupby('Arrival')['PAX'].apply(lambda x: x.tolist())

plt.figure(figsize=(12, 5))
plt.xlabel("PAX")
plt.ylabel("Counter")
plt.grid(True)
f_dep.hist(bins=8)
plt.savefig(str(os.getcwd()) + "\\images\\DepPax.png")

plt.figure(figsize=(12, 5))
plt.xlabel("PAX")
plt.ylabel("Counter")
plt.grid(True)
f_arr.hist(bins=8)
plt.savefig(str(os.getcwd()) + "\\images\\ArrPax.png")

from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')

fig = plt.figure(figsize=(5.5,5))
m = Basemap(projection='lcc', resolution=None,
            width=6E6, height=4E6,
            lat_0=40, lon_0=-95,)
m.etopo(scale=0.5, alpha=0.5)

dfd = df_train[['LatitudeDeparture', 'LongitudeDeparture']].drop_duplicates()
dfa = df_train[['LatitudeArrival', 'LongitudeArrival']].drop_duplicates()

for index,row in dfd.iterrows():
    x, y = m(row['LatitudeDeparture'], row['LongitudeDeparture'])
    plt.plot(x, y, 'ok', markersize=5)

    plt.text(x, y, str(df_train.at[index, 'Departure']), fontsize=10);

parallels = np.arange(20.,71,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(180., 300., 20.)
m.drawmeridians(meridians, labels=[True, False, False, True])
plt.show()
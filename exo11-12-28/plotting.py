import pandas as pd
from math import sqrt
import numpy as np
import statistics
import matplotlib.pyplot as plt

paxes = ['Pax0.csv', 'Pax1.csv', 'Pax2.csv', 'Pax3.csv', 'Pax4.csv', 'Pax5.csv', 'Pax6.csv', 'Pax7.csv']

confidence_level = [[0.5, 0.674],
                    [0.6, 0.842],
                    [0.7, 1.036],
                    [0.8, 1.282],
                    [0.9, 1.645],
                    [0.95, 1.960]]

# range = [WTD - (t_star*(std_wtd)/n, WTD + (t_star*(std_wtd)/n))]
'''
peirama1)
diasthma empistosynhs ana pax
meso oro ana deigma
afairesh oswn ektos orion
test
'''
def calc_range(df_pax, conf):
    sum_WTD = []
    sum_STDWTD = []
    for index, row in df_pax.iterrows():
        sum_WTD.append(row['WeeksToDeparture'])
        sum_STDWTD.append(row['std_wtd'])
    mean_WTD = sum(sum_WTD)/len(sum_WTD)
    std = []
    for number in sum_WTD:
        std.append((number-mean_WTD)**2)
    std = sum(std)/(len(std)-1)
    count = 0
    low_bar = mean_WTD - (conf[1]*(std/sqrt(len(sum_WTD))))
    high_bar = mean_WTD + (conf[1]*(std/sqrt(len(sum_WTD))))
    # print([low_bar, high_bar])
    for index, row in df_pax.iterrows():
        if (row['WeeksToDeparture'] < high_bar and row['WeeksToDeparture'] > low_bar):
            count += 1
    return count

def print_save_graphs(df_pax, confidence):
    for pax in range(len(paxes)):
        file = str(paxes[pax])
        df_pax = pd.read_csv(file)
        DateOfDeparture = df_pax['DateOfDeparture'].str.split('-')
        weeksToDeparture = df_pax['WeeksToDeparture']
        std_wtd = df_pax['std_wtd']
        plt.hist(weeksToDeparture, bins = 25)
       #plt.savefig('weeksToDeparture'+str(pax)+'_conf '+ confidence +'.png')
        plt.hist(std_wtd, bins = 25)
        plt.show()
        plt.show()
        plt.show()
        plt.show()
        plt.show()
       # plt.savefig('std_wtd'+str(pax)+'.png')


if __name__ == '__main__':
    count = 0
    for conf in confidence_level:
        for pax in paxes:
            df_pax = pd.read_csv(pax)
            #count += calc_range(df_pax, conf)
            print_save_graphs(df_pax,0)
       # print("Confidence: " + str(conf[0]) + ' : ' + str(count))






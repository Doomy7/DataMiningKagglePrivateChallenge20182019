import csv
import pandas as pd
import datetime
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import date
from keras import backend as K

#does everything
def masterManipulator(df_train, df_test, kaggle):
    droplist = [9, 10, 11]
    column_dict = {'0': 'DateOfDeparture', '1': 'Departure', '2': 'CityDeparture', '3': 'LongitudeDeparture',
                   '4': 'LatitudeDeparture',  '5': 'Arrival', '6': 'CityArrival', '7': 'LongitudeArrival',
                   '8': 'LatitudeArrival', '9': 'WeeksToDeparture', '10': 'std_wtd', '11': 'PAX'}
    df_train, df_test = datesEncoding(df_train), datesEncoding(df_test)
    df_train, df_test = trimester(df_train), trimester(df_test)
    droplist += [0]
    df_train, df_test = latlong_process(df_train), latlong_process(df_test)
    droplist += [3, 4, 7, 8]
    df_train, df_test = getCardinals(df_train), getCardinals(df_test)
    df_train, df_test = dropcolumns(droplist, column_dict, df_train, df_test, kaggle)
    df_train, df_test = lbencoder(df_train), lbencoder(df_test)
    df_train, df_test = onehot(df_train, df_test)
    return df_train, df_test

#prepare for kaggle test
def kaggletest():
    df_train = pd.read_csv("train.csv")
    return df_train, pd.read_csv("test.csv"), df_train[['PAX']], []

#prepare for local test
def localtest():
    df_train = pd.read_csv("train.csv")
    return train_test_split(df_train, df_train['PAX'], stratify=df_train['PAX'], test_size=0.1)

#split dateofDeparture & get day of week & week of year
def datesEncoding(dataframe):
    split_dates = dataframe['DateOfDeparture'].str.split('-', expand=True)
    split_dates.columns = ['FYear', 'FMonth', 'FDay']
    for index, row in split_dates.iterrows():
        split_dates.at[index, 'FD06'] = date(int(row['FYear']), int(row['FMonth']), int(row['FDay'])).weekday()
        dataframe.at[index, 'WeekOfYear'] = datetime.date(int(row['FYear']), int(row['FMonth']), int(row['FDay'])).isocalendar()[1]
    dataframe = pd.concat([split_dates, dataframe], axis=1)
    return dataframe

#build trimester
def trimester(dataframe):
    for index, row in dataframe.iterrows():
        if(int(row['FMonth']) > 8):
            dataframe.at[index, 'Trimester'] = 3
        elif(int(row['FMonth']) > 5):
            dataframe.at[index, 'Trimester'] = 2
        else:
            dataframe.at[index, 'Trimester'] = 1
    return dataframe

#get distance of Departure and arrival
def latlong_process(dataframe):
    for index, row in dataframe.iterrows():
        dataframe.at[index, 'Distance'] = str(math.hypot(row['LatitudeArrival'] - row['LatitudeDeparture'], row['LongitudeArrival'] - row['LongitudeDeparture']))
    return dataframe

#get cardinal (NE, NW, SE, SW)
def getCardinals(dataframe):
    for index, row in dataframe.iterrows():
        start, finish = cardinal_direction_simple(row)
        dataframe.at[index, 'Start'] = start
        dataframe.at[index, 'Finish'] = finish
    return dataframe

#cardinal helper
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

#drop not needed columns
def dropcolumns(droplist, column_dict, df_train, df_test, kaggle):
    if(kaggle):
        for column in droplist:
            df_train.drop(df_train[[column_dict.get(str(column))]], axis=1, inplace=True)
        for column in droplist:
            if(column != 11):
                df_test.drop(df_test[[column_dict.get(str(column))]], axis=1, inplace=True)
    else:
        for column in droplist:
            df_train.drop(df_train[[column_dict.get(str(column))]], axis=1, inplace=True)
            df_test.drop(df_test[[column_dict.get(str(column))]], axis=1, inplace=True)
    return df_train, df_test

#encode categorical or float numbers(ex. Distance)
def lbencoder(dataframe):
    le = LabelEncoder()
    le.fit(dataframe['Departure'])
    dataframe['Departure'] = le.transform(dataframe['Departure'])
    le.fit(dataframe['Arrival'])
    dataframe['Arrival'] = le.transform(dataframe['Arrival'])
    le.fit(dataframe['CityDeparture'])
    dataframe['CityDeparture'] = le.transform(dataframe['CityDeparture'])
    le.fit(dataframe['CityArrival'])
    dataframe['CityArrival'] = le.transform(dataframe['CityArrival'])
    le.fit(dataframe['Distance'])
    dataframe['Distance'] = le.transform(dataframe['Distance'])
    return dataframe

#one hot everything
def onehot(df_train, df_test):
    enc = OneHotEncoder(sparse=False)
    enc.fit(df_train)
    X_train = enc.transform(df_train)
    X_test = enc.transform(df_test)
    return X_train, X_test

#helper f1 calculation by Keras Backend
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#finalize (print f1 or generate prediction csv
def finalize(y_test, y_pred, kaggle):
    if(kaggle):
        with open('y_pred.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Id', 'Label'])
            for i in range(y_pred.shape[0]):
                writer.writerow([i, y_pred[i]])
    else:
        print(f1_score(y_test, y_pred, average='micro'))
import pandas as pd


def combineDepArr(df_train, df_test):
    deparr_train = df_train['Departure'] + df_train['Arrival']
    df_train.drop(df_train[['Departure']], axis=1, inplace=True)
    df_train.drop(df_train[['Arrival']], axis=1, inplace=True)
    df_train = pd.concat([deparr_train, df_train], axis=1)
    df_train.columns = ['DepArr', 'DateOfDeparture']
    deparr_test = df_test['Departure'] + df_test['Arrival']
    df_test.drop(df_test[['Departure']], axis=1, inplace=True)
    df_test.drop(df_test[['Arrival']], axis=1, inplace=True)
    df_test = pd.concat([deparr_test, df_test], axis=1)
    df_test.columns = ['DepArr', 'DateOfDeparture']
    return df_train, df_test


classWeight = mp.classWeight(y_train)

def classWeight(y_train):
    classWeight = (y_train.groupby('PAX').size()/y_train['PAX'].size).to_dict()
    leastK = min(classWeight, key=classWeight.get)
    leastV = classWeight[leastK]
    for key in classWeight.keys():
        classWeight[key] = (classWeight[key]/leastV)
    return classWeight


def latlong_comb(df_train, df_test):
    df_train['StartPos'] = abs(df_train['LatitudeDeparture'] + df_train['LongitudeDeparture'])
    df_train['EndPos'] = abs(df_train['LatitudeArrival'] + df_train['LongitudeArrival'])
    df_train.drop(df_train[['LatitudeDeparture']], axis=1, inplace=True)
    df_train.drop(df_train[['LongitudeDeparture']], axis=1, inplace=True)
    df_train.drop(df_train[['LatitudeArrival']], axis=1, inplace=True)
    df_train.drop(df_train[['LongitudeArrival']], axis=1, inplace=True)

    df_test['StartPos'] = abs(df_test['LatitudeDeparture'] + df_test['LongitudeDeparture'])
    df_test['EndPos'] = abs(df_test['LatitudeArrival'] + df_test['LongitudeArrival'])
    df_test.drop(df_test[['LatitudeDeparture']], axis=1, inplace=True)
    df_test.drop(df_test[['LongitudeDeparture']], axis=1, inplace=True)
    df_test.drop(df_test[['LatitudeArrival']], axis=1, inplace=True)
    df_test.drop(df_test[['LongitudeArrival']], axis=1, inplace=True)

    return df_train, df_test


def datesDistance(df_train, df_test):
    '''
    RESERVATION DATE TRAIN
    '''
    for index, row in df_train.iterrows():
        df_train.at[index, 'Wmean'] = int(math.modf(row['WeeksToDeparture'])[1])
        df_train.at[index, 'std'] = int(math.modf(row['std_wtd'])[1])

    '''
    RESERVATION DATE TEST
    '''
    for index, row in df_test.iterrows():
        df_test.at[index, 'Wmean'] = int(math.modf(row['WeeksToDeparture'])[1])
        df_test.at[index, 'std'] = int(math.modf(row['std_wtd'])[1])

    return df_train, df_test

def datesDistance(df_train, df_test):
    '''
    RESERVATION DATE TRAIN
    '''
    for index, row in df_train.iterrows():
        delta = timedelta(weeks=row['WeeksToDeparture'])
        df_train.at[index, 'Delta'] = str(delta.days)

    '''
    RESERVATION DATE TEST
    '''
    for index, row in df_test.iterrows():
        delta = timedelta(weeks=row['WeeksToDeparture'])
        df_test.at[index, 'Delta'] = str(delta.days)

    return df_train, df_test


def datesEncoding(df_train, df_test, combination):
    '''
    DATE PROCESSING
    '''

    '''
    FLIGHT DATE TRAIN
    '''
    Fdates_train = df_train[['DateOfDeparture']]
    split_dates_train = Fdates_train['DateOfDeparture'].str.split('-', expand=True)
    split_dates_train.columns = ['FYear', 'FMonth', 'FDay']
    split_dates_train['FD06'] = ""
    for index, row in split_dates_train.iterrows():
        split_dates_train.at[index, 'FD06'] = str(date(int(row['FYear']), int(row['FMonth']), int(row['FDay'])).weekday())

    '''
    FLIGHT DATE TEST
    '''
    Fdates_test = df_test[['DateOfDeparture']]
    split_dates_test = Fdates_test['DateOfDeparture'].str.split('-', expand=True)
    split_dates_test.columns = ['FYear', 'FMonth', 'FDay']
    split_dates_test['FD06'] = ""
    for index, row in split_dates_test.iterrows():
        split_dates_test.at[index, 'FD06'] = str(date(int(row['FYear']), int(row['FMonth']), int(row['FDay'])).weekday())

    '''
    SO MANY CHOISES
    '''
    df_train = pd.concat([split_dates_train, df_train], axis=1)
    df_test = pd.concat([split_dates_test, df_test], axis=1)

    if(combination > 0 and combination < 7):
        '''
        RESERVATION DATE TRAIN
        '''
        Rdates_train = pd.DataFrame({'Reservation': []})
        for index, row in df_train.iterrows():
            part_of_week, weeks = math.modf(row['WeeksToDeparture'])
            part_of_day, days = math.modf(7 * part_of_week)
            flightDay = date(int(row['FYear']), int(row['FMonth']), int(row['FDay']))
            Rdates_train.at[index, 'Reservation'] = str(flightDay - timedelta(weeks=weeks, days=days))
        Rdates_train = Rdates_train['Reservation'].str.split('-', expand=True)
        Rdates_train.columns = ['RYear', 'RMonth', 'RDay']
        Rdates_train['RD06'] = ""
        for index, row in Rdates_train.iterrows():
            Rdates_train.at[index, 'RD06'] = str(date(int(row['RYear']), int(row['RMonth']), int(row['RDay'])).weekday())

        '''
        RESERVATION DATE TEST
        '''
        Rdates_test = pd.DataFrame({'Reservation': []})
        for index, row in df_test.iterrows():
            part_of_week, weeks = math.modf(row['WeeksToDeparture'])
            part_of_day, days = math.modf(7 * part_of_week)
            flightDay = date(int(row['FYear']), int(row['FMonth']), int(row['FDay']))
            Rdates_test.at[index, 'Reservation'] = str(flightDay - timedelta(weeks=weeks, days=days))
        Rdates_test = Rdates_test['Reservation'].str.split('-', expand=True)
        Rdates_test.columns = ['RYear', 'RMonth', 'RDay']
        Rdates_test['RD06'] = ""
        for index, row in Rdates_test.iterrows():
            Rdates_test.at[index, 'RD06'] = str(date(int(row['RYear']), int(row['RMonth']), int(row['RDay'])).weekday())

        '''
        SO MANY CHOISES
        '''
        if(combination == 1):
            df_train = pd.concat([Rdates_train, df_train], axis=1)
            df_test = pd.concat([Rdates_test, df_test], axis=1)
        elif(combination == 2):
            df_train = pd.concat([Rdates_train['RD06'], df_train], axis=1)
            df_test = pd.concat([Rdates_test['RD06'], df_test], axis=1)
            df_train.drop(df_train[['FYear']], axis=1, inplace=True)
            df_test.drop(df_test[['FYear']], axis=1, inplace=True)
            df_train.drop(df_train[['FMonth']], axis=1, inplace=True)
            df_test.drop(df_test[['FMonth']], axis=1, inplace=True)
            df_train.drop(df_train[['FDay']], axis=1, inplace=True)
            df_test.drop(df_test[['FDay']], axis=1, inplace=True)
        elif(combination == 3):
            df_train = pd.concat([Rdates_train['RDay'], Rdates_train['RD06'], df_train], axis=1)
            df_test = pd.concat([Rdates_test['RDay'], Rdates_test['RD06'], df_test], axis=1)
            df_train.drop(df_train[['FYear']], axis=1, inplace=True)
            df_test.drop(df_test[['FYear']], axis=1, inplace=True)
            df_train.drop(df_train[['FMonth']], axis=1, inplace=True)
            df_test.drop(df_test[['FMonth']], axis=1, inplace=True)
        elif(combination == 4):
            df_train = pd.concat([Rdates_train['RMonth'], Rdates_train['RD06'], df_train], axis=1)
            df_test = pd.concat([Rdates_test['RMonth'], Rdates_test['RD06'], df_test], axis=1)
            df_train.drop(df_train[['FYear']], axis=1, inplace=True)
            df_test.drop(df_test[['FYear']], axis=1, inplace=True)
            df_train.drop(df_train[['FDay']], axis=1, inplace=True)
            df_test.drop(df_test[['FDay']], axis=1, inplace=True)
        elif(combination == 5):
            df_train = pd.concat([Rdates_train['RMonth'], Rdates_train['RDay'], Rdates_train['RD06'], df_train], axis=1)
            df_test = pd.concat([Rdates_test['RMonth'], Rdates_test['RDay'], Rdates_test['RD06'], df_test], axis=1)
            df_train.drop(df_train[['FYear']], axis=1, inplace=True)
            df_test.drop(df_test[['FYear']], axis=1, inplace=True)
        elif(combination == 6):
            df_train = pd.concat([Rdates_train['RYear'], Rdates_train['RMonth'], Rdates_train['RD06'], df_train], axis=1)
            df_test = pd.concat([Rdates_test['RYear'], Rdates_test['RMonth'], Rdates_test['RD06'], df_test], axis=1)
            df_train.drop(df_train[['FDay']], axis=1, inplace=True)
            df_test.drop(df_test[['FDay']], axis=1, inplace=True)


    return df_train, df_test


    def reduceOversample(df_train):
        new_df_train = pd.Dataframe()
        for pax in range(0, 8):
            df_pax = df_train.loc[df_train['PAX'] == pax]
            if(df_pax.size() > 1000):
                df_pax = df_pax.sample(df_pax.size() - 500)
                new_df_train = pd.concat([df_train, pax], axis=1)
        return df_train



    def doStatistics(df_train, y_train, conf):
        df_pax0 = df_train.loc[df_train['PAX'] == 0]
        df_pax1 = df_train.loc[df_train['PAX'] == 1]
        df_pax2 = df_train.loc[df_train['PAX'] == 2]
        df_pax3 = df_train.loc[df_train['PAX'] == 3]
        df_pax4 = df_train.loc[df_train['PAX'] == 4]
        df_pax5 = df_train.loc[df_train['PAX'] == 5]
        df_pax6 = df_train.loc[df_train['PAX'] == 6]
        df_pax7 = df_train.loc[df_train['PAX'] == 7]
        paxes = [df_pax0, df_pax1, df_pax2, df_pax3, df_pax4, df_pax5, df_pax6, df_pax7]
        conf_range = {}
        count = 0
        ind = 0
        for df_pax in paxes:
            sum_WTD = []
            # sum_STDWTD = []
            for index, row in df_pax.iterrows():
                sum_WTD.append(row['WeeksToDeparture'])
                # sum_STDWTD.append(row['std_wtd'])
            mean_WTD = sum(sum_WTD)/len(sum_WTD)
            std = []
            for number in sum_WTD:
                std.append((number-mean_WTD)**2)
            std = sum(std)/(len(std)-1)
            low_bar = mean_WTD - (conf[1]*(std/math.sqrt(len(sum_WTD)))) - 4
            high_bar = mean_WTD + (conf[1]*(std/math.sqrt(len(sum_WTD)))) + 4
            # print([low_bar, high_bar])
            for index, row in df_pax.iterrows():
                if (row['WeeksToDeparture'] < high_bar and row['WeeksToDeparture'] > low_bar):
                    count += 1
            conf_range[ind] = [low_bar, high_bar]
            ind += 1
        rows_to_drop = []
        for index, row in df_train.iterrows():
            if(row['PAX'] == 0):
                if(row['WeeksToDeparture'] < conf_range[0][0] or row['WeeksToDeparture'] > conf_range[0][1]):
                    rows_to_drop.append(index)
            elif(row['PAX'] == 1):
                if(row['WeeksToDeparture'] < conf_range[1][0] or row['WeeksToDeparture'] > conf_range[1][1]):
                    rows_to_drop.append(index)
            elif(row['WeeksToDeparture'] == 2):
                if(row['WeeksToDeparture'] < conf_range[2][0] or row['WeeksToDeparture'] > conf_range[2][1]):
                    rows_to_drop.append(index)
            elif(row['WeeksToDeparture'] == 3):
                if(row['WeeksToDeparture'] < conf_range[3][0] or row['WeeksToDeparture'] > conf_range[3][1]):
                    rows_to_drop.append(index)
            elif(row['WeeksToDeparture'] == 4):
                if(row['WeeksToDeparture'] < conf_range[4][0] or row['WeeksToDeparture'] > conf_range[4][1]):
                    rows_to_drop.append(index)
            elif(row['WeeksToDeparture'] == 5):
                if(row['WeeksToDeparture'] < conf_range[5][0] or row['WeeksToDeparture'] > conf_range[5][1]):
                    rows_to_drop.append(index)
            elif(row['WeeksToDeparture'] == 6):
                if(row['WeeksToDeparture'] < conf_range[6][0] or row['WeeksToDeparture'] > conf_range[6][1]):
                    rows_to_drop.append(index)
            elif(row['WeeksToDeparture'] == 7):
                if(row['WeeksToDeparture'] < conf_range[7][0] or row['WeeksToDeparture'] > conf_range[7][1]):
                    rows_to_drop.append(index)
        for row in rows_to_drop:
            df_train = df_train.drop([row])
            y_train = y_train.drop([row])
        return df_train, y_train


    def doStatistics2(df_train, df_test):

        for index, row in df_train.iterrows():
            df_train.at[index, 'Low_bar'] = round(float(row['WeeksToDeparture'] - 1.96*(row['std_wtd']/10)), 1)
            df_train.at[index, 'High_bar'] = round(float(row['WeeksToDeparture'] + 1.96*(row['std_wtd']/10)), 1)
        for index, row in df_test.iterrows():
            df_test.at[index, 'Low_bar'] = round(float(row['WeeksToDeparture'] - 1.96*(row['std_wtd']/10)), 1)
            df_test.at[index, 'High_bar'] = round(float(row['WeeksToDeparture'] + 1.96*(row['std_wtd']/10)), 1)
        return df_train, df_test

        #if int(row['FDay']) > 15:
        #    split_dates.at[index, 'BiWeekly'] = int(row['FMonth'])*2
        #else:
        #    split_dates.at[index, 'BiWeekly'] = int(row['FMonth'])*2-1
statistics = 1
confidence_level = [[0.5, 0.674],
                    [0.6, 0.842],
                    [0.7, 1.036],
                    [0.8, 1.282],
                    [0.9, 1.645],
                    [0.95, 1.960]]

if(statistics):
    # df_train, y_train = doStatistics(df_train, y_train, confidence_level[5])
    df_train, df_test = doStatistics2(df_train, df_test)
    pass


    le.fit(df_train['Low_bar'])
    df_train['Low_bar'] = le.transform(df_train['Low_bar'])
    le.fit(df_test['Low_bar'])
    df_test['Low_bar'] = le.transform(df_test['Low_bar'])

    le.fit(df_train['High_bar'])
    df_train['High_bar'] = le.transform(df_train['High_bar'])
    le.fit(df_test['High_bar'])
    df_test['High_bar'] = le.transform(df_test['High_bar'])


def flightDay(df_train, df_test):

    for index, row in df_train.iterrows():
        part_of_week, weeks = math.modf(row['WeeksToDeparture'])
        part_of_day, days = math.modf(7 * part_of_week)
        flightDay = date(int(row['FYear']), int(row['FMonth']), int(row['FDay']))
        df_train.at[index, 'Reservation'] = str(flightDay - timedelta(weeks=weeks, days=days))
    df_train[['RYear', 'RMonth', 'RDay']] = df_train['Reservation'].str.split('-', expand=True)
    df_train['RD06'] = ""
    for index, row in df_train.iterrows():
        df_train.at[index, 'RD06'] = str(date(int(row['RYear']), int(row['RMonth']), int(row['RDay'])).weekday())

    for index, row in df_test.iterrows():
        part_of_week, weeks = math.modf(row['WeeksToDeparture'])
        part_of_day, days = math.modf(7 * part_of_week)
        flightDay = date(int(row['FYear']), int(row['FMonth']), int(row['FDay']))
        df_test.at[index, 'Reservation'] = str(flightDay - timedelta(weeks=weeks, days=days))
    df_test[['RYear', 'RMonth', 'RDay']] = df_test['Reservation'].str.split('-', expand=True)
    df_test['RD06'] = ""
    for index, row in df_test.iterrows():
        df_test.at[index, 'RD06'] = str(date(int(row['RYear']), int(row['RMonth']), int(row['RDay'])).weekday())

    df_train.drop(df_train[['Reservation']], axis=1, inplace=True)
    df_test.drop(df_test[['Reservation']], axis=1, inplace=True)
    #df_train.drop(df_train[['RYear']], axis=1, inplace=True)
    #df_test.drop(df_test[['RYear']], axis=1, inplace=True)
    #df_train.drop(df_train[['RDay']], axis=1, inplace=True)
    #df_test.drop(df_test[['RDay']], axis=1, inplace=True)
    return df_train, df_test



def day_by_day(dataframe):
    for index, row in dataframe.iterrows():
        dataframe.at[index, "DD"] = int(row['FMonth'])*int(row['FDay'])
    return dataframe


def weekDif(dataframe):
    for index, row in dataframe.iterrows():
        if int(row['RYear']) < int(row['FYear']):
            if(52 + row['WeekOfYear'] - row['RWeekOfYear'] > 52):
                dataframe.at[index, 'WeekDif'] = row['WeekOfYear'] - row['RWeekOfYear']
            else:
                dataframe.at[index, 'WeekDif'] = 52 + row['WeekOfYear'] - row['RWeekOfYear']
        else:
            if(row['WeekOfYear'] - row['RWeekOfYear'] < 0):
                if(row['WeekOfYear']) == 1.0:
                    dataframe.at[index, 'WeekDif'] = 52 - row['RWeekOfYear']
                elif row['RWeekOfYear'] == 52.0:
                    dataframe.at[index, 'WeekDif'] = row['WeekOfYear']
            else:
                dataframe.at[index, 'WeekDif'] = row['WeekOfYear'] - row['RWeekOfYear']
    dataframe.drop(dataframe[['RYear']], axis=1, inplace=True)
    dataframe.drop(dataframe[['RMonth']], axis=1, inplace=True)
    dataframe.drop(dataframe[['RDay']], axis=1, inplace=True)
    dataframe.drop(dataframe[['RD06']], axis=1, inplace=True)
    dataframe.drop(dataframe[['RWeekOfYear']], axis=1, inplace=True)
    return dataframe

def semester(dataframe):
    for index, row in dataframe.iterrows():
        if(int(row['FMonth']) > 6):
            dataframe.at[index, 'Semester'] = 2
        else:
            dataframe.at[index, 'Semester'] = 1
    return dataframe


def quarter(datafgrame):
    for index, row in datafgrame.iterrows():
        if(int(row['FMonth']) > 9):
            datafgrame.at[index, 'Quarter'] = 4
        elif(int(row['FMonth']) > 6):
            datafgrame.at[index, 'Quarter'] = 3
        elif(int(row['FMonth']) > 3):
            datafgrame.at[index, 'Quarter'] = 2
        else:
            datafgrame.at[index, 'Quarter'] = 1
    return datafgrame


def rDatesEncoding(dataframe):
    for index, row in dataframe.iterrows():
        part_of_week, weeks = math.modf(row['WeeksToDeparture'])
        part_of_day, days = math.modf(7 * part_of_week)
        flightDay = date(int(row['FYear']), int(row['FMonth']), int(row['FDay']))
        dataframe.at[index, 'Reservation'] = str(flightDay - timedelta(weeks=weeks, days=days))
    split_Rdates = dataframe['Reservation'].str.split('-', expand=True)
    split_Rdates.columns = ['RYear', 'RMonth', 'RDay']
    split_Rdates['RD06'] = ""
    for index, row in split_Rdates.iterrows():
        split_Rdates.at[index, 'RD06'] = str(date(int(row['RYear']), int(row['RMonth']), int(row['RDay'])).weekday())
        #dataframe.at[index, 'RWeekOfYear'] = datetime.date(int(row['RYear']), int(row['RMonth']), int(row['RDay'])).isocalendar()[1]
    dataframe = pd.concat([split_Rdates, dataframe], axis=1)
    dataframe.drop(dataframe[['Reservation']], axis=1, inplace=True)
    #dataframe.drop(dataframe[['RYear']], axis=1, inplace=True)
    #dataframe.drop(dataframe[['RMonth']], axis=1, inplace=True)
    #dataframe.drop(dataframe[['RDay']], axis=1, inplace=True)
    #dataframe.drop(dataframe[['RD06']], axis=1, inplace=True)
    return dataframe

def Standardize(df_train, df_test):
    scaler = StandardScaler()
    df_train["WeeksToDeparture"] = df_train["WeeksToDeparture"].astype('int')
    df_train["std_wtd"] = df_train["std_wtd"].astype('float')
    df_test["WeeksToDeparture"] = df_test["WeeksToDeparture"].astype('float')
    df_test["std_wtd"] = df_test["std_wtd"].astype('float')
    scaler.fit(df_train[["WeeksToDeparture", "std_wtd"]])
    df_train[["WeeksToDeparture", "std_wtd"]] = scaler.transform(df_train[["WeeksToDeparture", "std_wtd"]])
    scaler.fit(df_test[["WeeksToDeparture", "std_wtd"]])
    df_test[["WeeksToDeparture", "std_wtd"]] = scaler.transform(df_test[["WeeksToDeparture", "std_wtd"]])
    return df_train, df_test



def flight_direction(dataframe):
    for index, row in dataframe.iterrows():
        if row['Start'][0] == row['Finish'][0]:
            if row['Start'] == 'CC':
                dataframe.at[index, 'Direction'] = 'E'
            else:
                dataframe.at[index, 'Direction'] = row['Finish'][1]
        elif row['Start'][1] == row['Finish'][1]:
            if row['Finish'][0] == 'C':
                if row['Start'][0] == 'N':
                    dataframe.at[index, 'Direction'] = 'S'
                else:
                    dataframe.at[index, 'Direction'] = 'N'
            else:
                dataframe.at[index, 'Direction'] = row['Finish'][0]
        else:
            if row['Start'][0] == 'N':
                dataframe.at[index, 'Direction'] = 'SE'
            elif row['Start'][0] == 'S':
                dataframe.at[index, 'Direction'] = 'NE'
            elif row['Start'] == 'CC':
                dataframe.at[index, 'Direction'] = row['Finish']
            elif row['Start'][0] == 'C':
                if row['Finish'][0] == 'N':
                    dataframe.at[index, 'Direction'] = 'NE'
                elif row['Finish'][0] == 'S':
                    dataframe.at[index, 'Direction'] = 'SE'
    #dataframe.drop(dataframe[['Start']], axis=1, inplace=True)
    #dataframe.drop(dataframe[['Finish']], axis=1, inplace=True)
    return dataframe



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

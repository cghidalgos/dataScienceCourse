import numpy as np
import os
import re
import folium
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from folium.plugins import FastMarkerCluster
from matplotlib.pyplot import figure
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.pyplot import figure
import math


# You will call this function multiple times in this notebook. Think of this function as a tool which will take the data,
# fit into the Linear Regression Model and Evaluate

def missing_values_imputation_mean(train_df, test_df, target):
    '''
    This function will take the features(x), the target(y) and the model name (Linear Regression)
    and will fit the data into the model (train your data using Linear Regression) 
    and Evaluate by returning the mean squared error and the mean absolute error 
    '''
    
    y_train = train_df[target]
    y_test = test_df[target]
    
    y_pred = np.array([np.mean(y_train)] * len(y_test))
    
    return {"mse": mean_squared_error(y_pred, y_test),"mae": mean_absolute_error(y_pred, y_test)}

def missing_values_imputation_mean_by_station(train_df, test_df, target):
    '''
    This function will take the features(x), the target(y) and the model name (Linear Regression)
    and will fit the data into the model (train your data using Linear Regression) 
    and Evaluate by returning the mean squared error and the mean absolute error 
    '''
    y_test = test_df[target]

    mean_by_station = train_df.groupby(['Station']).agg(({target: 'mean'}))
    map_station_mean = {}

    for index, row in mean_by_station.iterrows():
        map_station_mean[row.name] = row[0]

    def predictor(x):
        return map_station_mean[x['Station']]

    y_pred = test_df.apply(predictor, axis=1).values
    
    return {"mse": mean_squared_error(y_pred, y_test),"mae": mean_absolute_error(y_pred, y_test)}


def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'S' or direction == 'W':
        dd *= -1
    return dd;


def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]


def parse_dms(coor):
    parts = re.split('[^\d\w]+', coor)
    dec_coor = dms2dd(parts[0], parts[1], float(parts[2]+'.'+parts[2]), parts[4])

    return dec_coor


def replace24(datetimex):
    return datetimex.replace('24:00', '00:00')


def plot_features(df, feature_names):
    # declarando un objeto tipo Figura para desarrollar los subplots
    fig = plt.figure(figsize=(20, 10))

    x = 1

    for column in feature_names:
        ax = fig.add_subplot(3,5,x)
        df[[column]].plot(kind='hist', ax=ax, rwidth=1)
        x = x + 1
        

def color_producer(pm):
    if pm < 12.0:
        return 'green'
    elif pm < 35.0:
        return 'yellow'
    elif pm < 55.4:
        return 'orange'
    elif pm < 150:
        return 'red'
    else:
        return 'purple'
    
def leaflet_plot(data):
    '''
    Create a plot to visualize 2 set of geo points. The popup will show a scatter with the average hourly emisions
    '''
    data = data[['Latitud', 'Longitud', 'PM2.5', 'Station', 'hour']]
    data2 = data.groupby(['Station', 'hour']).agg(({'PM2.5': 'mean'}))

    grouped_data = {}
    for index, row in data2.iterrows():
        if index[0] in grouped_data:
            grouped_data[index[0]][index[1]] = row[0]
        else:
            grouped_data[index[0]] = [0] * 24
            grouped_data[index[0]][index[1]] = row[0]

    for key in grouped_data:
        plt.plot(list(range(0, 24)), grouped_data[key], '-o')
        plt.title(f'Station {key} avg. PM2.5 / hour')
        plt.xlabel('hour')
        plt.ylabel('Avg. PM2.5') 
        plt.savefig(f'img/tmp/{key}.png')
        plt.clf()

    #     data2 = temp_df.groupby('Station').agg(({'PM2.5': 'mean', 'Latitud': 'min', 'Longitud': 'min'}))
    data2 = data.groupby('Station').agg(({'PM2.5': 'mean', 'Latitud': 'min', 'Longitud': 'min'}))
    data2 = np.array([data2['Latitud'].values, data2['Longitud'].values, data2['PM2.5'].values, data2.index.values]).T
    #print((data[0].Latitud, data[0].Longitud))
    map3 = folium.Map(location=[data2[0][0], data2[0][1]], tiles='openstreetmap', zoom_start=11)
    
    fg = folium.FeatureGroup(name="My Map")
    for lt, ln, pol, station in data2:
        fg.add_child(folium.CircleMarker(location=[lt, ln], radius = 15, popup=f"<img src='img/tmp/{station}.png'>",
        fill_color=color_producer(pol), color = '', fill_opacity=0.5))
        map3.add_child(fg)
    return map3


def draw_example(sample, missing_index, interpolate=True, title=''):
    missing = missing_index
    missing_before_after = [missing[0]-1]+missing+ [missing[-1]+1]

    example1 = sample.copy()
    example1.loc[example1['hour'].isin(missing),'PM2.5'] = float('NaN')
    plt.plot(missing_before_after,  sample[sample['hour'].isin(missing_before_after)]['PM2.5'] , 'r--o')

    if interpolate:
        example1['newPM2.5'] = example1['PM2.5'].interpolate(method='linear')
        plt.plot(missing_before_after, example1[example1['hour'].isin(missing_before_after)]['newPM2.5'], 'g--o')
        
    plt.plot(example1['hour'], example1['PM2.5'], '-*')
    
    plt.xlabel('Hour')
    plt.ylabel('PM2.5')
    plt.title(title)

def get_size_down_periods(df):
    distribution = [0] * 4000
    x = []
    i = -1
    total_missing = 0
    count = 0
    for row in df['PM2.5'].values:
        if math.isnan(row):
            total_missing += 1
            if i == 0:
                count = 1
                i = 1
            else:
                count += 1
        else:
            try:
                if count > 0:
                    distribution[count] += 1 
                    x.append(count)
            except:
                print(count)
            i = 0
            count = 0

    distribution[0] = df['PM2.5'].shape[0] - total_missing
    return distribution
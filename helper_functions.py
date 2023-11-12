#---------------------------------------------------------------
#
# Author: Cezary Januszek
# Created: Tuesday, November 7th 2023
#
#----------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn
import datetime


#----------------------------------------------------------------
# Data pre-processing functions
#----------------------------------------------------------------

def convert_to_time(time):
    '''
    Function to convert float time to HH:MM format
    '''
    if pd.isnull(time):
        return np.nan
    else:
        if time >= 2400:
            time = time % 2400
        time_string = '{0:04}'.format(int(time))
        hour_min = datetime.time(int(time_string[:2]), int(time_string[2:]))
        return hour_min
    
def combine_date_time(x, date, time):
    '''
    Combine date and time to create full datetime object
    '''
    if pd.isnull(x[date]) or pd.isnull(x[time]):
        return np.nan
    else:
        return datetime.datetime.combine(x[date], x[time])
    
def get_arrival_date(x):
    '''
    Compute the arrival date based on departure and arrival time.
    If the arrival time is smaller than the departure time then the arrival is on the next day
    '''
    if x['SCHEDULED_DEPARTURE'] > x['SCHEDULED_ARRIVAL']:
        return x['DATE'] + datetime.timedelta(days=1)
    else:
        return x['DATE']
    

def preprocess_for_regression(df):
    return 0
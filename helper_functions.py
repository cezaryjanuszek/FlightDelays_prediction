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

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    


def clf_evaluate_metrics(clf, x_test, y_test):
    '''
    Compute classification model predictions and probabilities and evaluate model's performance
    '''
    y_pred_proba = clf.predict_proba(x_test)
    loss = log_loss(y_test, y_pred_proba)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('Cross-entropy loss = {:.3f}'.format(loss))
    print('Accuracy = {:.3f}'.format(acc))
    print('Precision = {:.3f}'.format(precision))
    print('Recall = {:.3f}'.format(recall))
    print('F1-score = {:.3f}'.format(f1))

    conf_matrix = confusion_matrix(y_test, y_pred, labels=['ON-TIME', 'DELAY'])

    sns.heatmap(conf_matrix, labels=conf_matrix.index)

    return y_pred, y_pred_proba
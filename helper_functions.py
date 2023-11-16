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

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
    

#----------------------------------------
# Functions for prediction model
#----------------------------------------


def cyclical_encode(data, col, max_val):
    '''
    Encoding based on the idea that dates and times are cyclical and hours like 23:00 and 00:00 are closer to each other than if assumed linear
    The use of sine and cosine allows to put this time/date points on a circle representing their cyclical structure
    '''
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def set_delay_class(x):
    '''
    Custom label encoder for target variable - FLIGHT DELAY
    '''
    if x <= 15:
        # No delay
        return 0
    elif x > 15 and x <= 45:
        # Delay
        return 1
    elif x > 45 and x <= 120:
        # Important delay
        return 2
    else:
        # Big delay
        return 3


def clf_evaluate_metrics(clf, x_test, y_test):
    '''
    Compute classification model predictions and probabilities and evaluate model's performance
    '''
    y_pred_proba = clf.predict_proba(x_test)
    loss = log_loss(y_test, y_pred_proba)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')

    print('Cross-entropy loss = {:.3f}'.format(loss))
    print('Accuracy = {:.3f}'.format(acc))
    print('Precision = {:.3f}'.format(precision))
    print('Recall = {:.3f}'.format(recall))
    print('F1-score = {:.3f}'.format(f1))
    print('ROC-AUC = {:.3f}'.format(roc_auc))

    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize='true'), index=['No delay', 'Delay', 'Important delay', 'Big delay'], columns=['No delay', 'Delay', 'Important delay', 'Big delay'])
    
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap='vlag', fmt='.4f')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()

    return y_pred, y_pred_proba
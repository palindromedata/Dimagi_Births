#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:44:17 2018

@author: ksharpey
"""

import os as os
import numpy as np
import pandas as pd
import seaborn as sns
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

#INPUT_FILE = 'Outputs/preg cases period table trainSet 1 2018-02-06 - inc dob_output_20180226-16h06_EFC - days clean.csv'
INPUT_FILE = config['CAKE']['INPUT_FILE']
MOVING_AVG = config.getint('CAKE','MOVING_AVG')
MOVING_AVG_COL = str(MOVING_AVG) +'m mavg Monthly Total'
TARGETS = config.get('CAKE','TARGETS').split(',')
VERBOSE = config.getboolean('DEFAULT','VERBOSE')

# INITIATE
unified_view = pd.read_csv(INPUT_FILE)
df = unified_view 
unified_view.shape
df.shape
df['date'] = pd.to_datetime(df[config['CAKE']['TIMESERIES_BASEDATE_FIELD']], dayfirst=True, errors='coerce')#.astype("datetime64") errors='coerce'
#df = df[df['date'] >= '2013-06-01'] #9170-6221, lose ~3k, lets just keep it so the moving average is in
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['date'].value_counts().sum()  #only 6699 have recorded birth  6699/9170 73% have birth data
# CLEAN out messy first periods

for i in TARGETS:
    target = i
    x = df[['_id','month','year',target]]
    x = x.rename(columns={'_id':'all'})

    y = x.groupby(by=['year', 'month']).count()
    y[target] = x.groupby(by=['year', 'month']).sum()
    y['other'] = y['all']- y[target]
    y[MOVING_AVG_COL] = y['all'].rolling(window=MOVING_AVG).mean()
    y[[target,'other']].plot(kind='bar', stacked = True, figsize = (18,5), grid=True, rot=90)
    y[MOVING_AVG_COL].plot(color = 'black',linewidth=2.0,use_index = True, grid=True, rot=90, legend=True)

    y['other '+ str(MOVING_AVG)+'m mavg'] = y['other'].rolling(window=MOVING_AVG).mean()
    y['other '+ str(MOVING_AVG)+'m mavg'].plot(color = 'orange',linewidth=2.0,use_index = True, legend=True)
    y['target monthly%'] = y[target]/y['all']
    z = y['target monthly%']*100#*y['all'].median()
    z.plot(color = 'red',linewidth=2.0,use_index = True, legend=True, grid=True, rot=90,) 

    if VERBOSE:
#        print(i)
#        print(unified_view[target].value_counts())
        print("Mean for all data: ", unified_view[target].value_counts()[1]/len(unified_view), " are ", target)
        print("Mean of monthly means: ", y['target monthly%'].describe()[1])
        print("Median of monthly means: ", y['target monthly%'].describe()[5])

# =============================================================================
# TESTS        
# =============================================================================
#quick tests:
y['EN_PLACE_BIRTH_in_hospital'].sum()   #5121 5121+1578 = 6699, 5121/6699 = 76% incidence, meanwhile 5121/9037 ~ 57%
y['other'].sum()                #1578
len(unified_view['OC_date_birth_vip'])
len(unified_view[unified_view['OC_date_birth_vip']!="0"])

df['date'].value_counts().plot(figsize = (18,5))
x = df[['OC_MIN_timeperiodStart','OC_date_birth_vip']]          #df.columns.values 

df['OC_SUM_Number pregnancy visits'].describe()
df['OC_SUM_Number pregnancy visits'].sum()
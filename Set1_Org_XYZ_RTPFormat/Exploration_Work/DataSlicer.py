#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:31:35 2018

@author: ksharpey
"""

# =============================================================================
# DATA SLICER

#- read in an EFC file
#- filter by (date) on (filter_field)
#- write to file
# =============================================================================

import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

INPUT_FILE = config['DEFAULT']['INPUT_FILE']
FILTER_FIELD = 'OC_MIN_timeperiodStart'
date_filter = '2013-06-01'

df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df[FILTER_FIELD], dayfirst=True, errors='coerce')#.astype("datetime64") errors='coerce'

# =========== SLICE ==================================================================
x = df['date']>= date_filter      
x.sum()

y = df[df['date']>=date_filter]
y.shape
y.drop('date', axis=1, inplace=True)
# =========== Export ==================================================================
output_file_name = INPUT_FILE.rstrip('.csv') + '_after_'+date_filter+'.csv'
y.to_csv(output_file_name)
print(len(df), " filtered to ", len(y), " rows written to: ", output_file_name)

#TEST

y['date'].sort_values()
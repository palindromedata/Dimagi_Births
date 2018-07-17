#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:05:57 2018

@author: ksharpey
"""

# =============================================================================
# # a script for transposing Neal's simpler pregnancy data
# # The main goal is to produce a 'unified view' of the input data by:
#     1. finding all the rows per case
#     2. Creating realtive time periods (RTP) for the case
#     3. populating RTP columns of all dynamic observations
#     4. Copying static attributes
#
#    not doing: auto calculating what is static and what is dynamic
# =============================================================================

import pandas as pd
import numpy as np
import os as os
import seaborn as sns
#import my_functions as mf

# PARAMATERS
input_file = 'Flat_File.csv'
dynamic_col_names = ['openedInPeriod', 'closedInPeriod', 'anc_2',
        'anc_3', 'anc_4', 'tetanus_1', ] 
static_col_names = ['place_birth', 'type_delivery']
uique_id_col_name = 'case'
output_file_name = 'output5.csv'
# INITIATE

#clear
os.getcwd()
df = pd.read_csv(input_file)                              #original dataframe
new_df = pd.DataFrame(df.loc[:,uique_id_col_name].unique())          #transformed dataframe, initialised with the unique IDS
new_df.columns = ['_id']                                  #rename the column  

#prep DF with extra columns
num_TP = df[uique_id_col_name].value_counts()
max_TP = len(num_TP.value_counts())                       #135 have one visit  ,12 max visits


for AT_iterator in range(0,len(dynamic_col_names)):
    for n in range(0,max_TP):
        var = dynamic_col_names[AT_iterator]                           
        col = var +'-RTP'+ str(n+1)
        new_df.insert(len(new_df.columns),col,"")

#   Get a list of the unqiue case IDs
case_ids = pd.Series(df.loc[:,uique_id_col_name].unique())           #returns a series of unique case ids - 437

# LOOP i - go through each case
for i in range(0,len(case_ids)):                            #437 - iterates through all unique cases
    #   Return only the rows for a case id
    current_case_rows = df.loc[df[uique_id_col_name] == case_ids[i]]   #returns a pd.DataFrame of all the rows for current case 
    current_case_rows = current_case_rows.sort_values('timeperiod',ascending=True)       #order them by timeperiod 
    current_case_rows = current_case_rows.reset_index()     #reset the indexes to 0,1,2...
    relativeTP = len(current_case_rows)                     #e.g. 7 for 'c1', 2 for 'c2'
        
    #LOOP AT - go through each attribute
    for AT_iterator in range(0,len(dynamic_col_names)):
        var = dynamic_col_names[AT_iterator]                          #e.g. For ANC2 update each ANC val
        
        #LOOP TP - go through each TimePeriod
        for TP_iterator in range(1,relativeTP+1):                         #iterates through all TP
            col = var +'-RTP'+ str(TP_iterator)
            new_df.loc[new_df['_id']==case_ids[i], col] = current_case_rows.loc[TP_iterator-1,var]

    #LOOP STATIC - copy over static attributes just as single colums
    for j in range(0,len(static_col_names)):
        new_df.loc[new_df['_id']==case_ids[i], static_col_names[j]] = current_case_rows.loc[0,static_col_names[j]]
        

#Return result
new_df.to_csv(output_file_name)
print(output_file_name,' saved to', os.getcwd())

# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# def value_count_pct(input_df, column_name=0):
#     input_df = pd.DataFrame(input_df)
#     x = pd.DataFrame(input_df[column_name].value_counts())
#     x = x.rename(columns={ x.columns[0]: 'count' })
#     x.insert(len(x.columns),'pct',x['count']/len(input_df))
#     x = x.sort_index()
#     return(x)
# =============================================================================

# =============================================================================
# STATS ZONE
# =============================================================================
# =============================================================================
# 
# print('STATS start here')
# a = pd.DataFrame(data=[1,1,3,4,5], index=['a', 'b', 'c', 'd', 'e'])
# x = value_count_pct(a)
for AT_iterator in range(0,len(static_col_names)):
        var = static_col_names[AT_iterator]  
        print(new_df[var].value_counts())
# =============================================================================
len(new_df)
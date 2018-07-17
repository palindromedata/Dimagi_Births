#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:46:31 2018

@author: ksharpey
"""
# =============================================================================
# This script 
#  1 - Produces extra Engineered Features (EF) on the unified view
#  2 - Filters known outliers, cleans dates
#  3 - exports clean & engineered to output folder
#  
#  This script does not: correlation, regreression
# =============================================================================

import os as os
import pandas as pd
import seaborn as sns
import datetime
from datetime import datetime
from dateutil import relativedelta
from dateutil.relativedelta import relativedelta
import calendar
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
os.getcwd()

#constants
INPUT_FILE = config['EFC']['INPUT_FILE']

UNIQUE_ID_COL_NAME = '_id'
STATIC_INPUT_COLS = ['IN_EDD', 'IN_age', 'IN_previous_pregnancies', 'IN_living_children'] #4
DYNAMIC_COL_NAMES = ['openedInPeriod', 'closedInPeriod', 'anc_2', 'anc_3', 'anc_4', 'tetanus_1', 'takes_nutrition'] #7
OUTCOME_COLS = [
        'OC_date_birth_vip', 'OC_anc_2_date', 'OC_place_birth',
       'OC_type_delivery', 'OC_maternal_outcome', 'OC_pregnancy_outcome',
       'OC_birth_delivery_conducted_by', 'OC_pre_term', 'OC_birth_weight',
       'OC_MAX_Months Til EDD', 'OC_MIN_timeperiodStart']
OUTCOME_FACTOR_COLS = [
       'OC_place_birth',
       'OC_type_delivery', 'OC_maternal_outcome', 'OC_pregnancy_outcome',
       'OC_birth_delivery_conducted_by', 'OC_pre_term', 'OC_birth_weight']
ENGINEERED_COL =[
        'EF_tot_TP',#'EF_RTP_CLOSED',
        'EF_RTP_of_birth', 'EF_DIFF_EDD_DOB', 'EF_DOW_DOB', 'EF_DOW_EDD',
        'EF_Maternal_Age_Group','EF_Weeks_Preg_at_Presentation']

ABS_FIRST_TP = datetime.strptime(str('01/01/2012'), '%m/%d/%Y') 
#ABS_FIRST_TP.weekday()
#now = datetime.datetime.now()
#months_since = relativedelta.relativedelta(now, ABS_FIRST_TP)
#months_since.months
#test6months_add = ABS_FIRST_TP + relativedelta(months=+6)

# =============================================================================
# Prep df and output df
# =============================================================================
os.getcwd()
unified_view = pd.read_csv(INPUT_FILE)
new_df = unified_view

# =============================================================================
# CLEAN DATES
# =============================================================================
new_df['IN_EDD'] = pd.to_datetime(new_df['IN_EDD'].dropna())
new_df['OC_MIN_timeperiodStart'] = pd.to_datetime(new_df['OC_MIN_timeperiodStart'].dropna())
#dobs = new_df[new_df['OC_date_birth_vip']!='---']['OC_date_birth_vip'].dropna().astype('datetime64[ns]')
dobs = pd.to_datetime(new_df[new_df['OC_date_birth_vip']!='---']['OC_date_birth_vip'].dropna())

# =============================================================================
# HOT ENCODING
# https://lukesingham.com/whos-going-to-leave-next/
# =============================================================================
for i in range(0, len(OUTCOME_FACTOR_COLS)): 
    new_df[OUTCOME_FACTOR_COLS[i]] = 'EN_'+OUTCOME_FACTOR_COLS[i].upper()[3:]+'_' + new_df[OUTCOME_FACTOR_COLS[i]].astype(str) 
    # generate dummies column encoding and join
    y = pd.get_dummies(new_df[OUTCOME_FACTOR_COLS[i]])  
    new_df = new_df.join(y)  

# =============================================================================
# EF
# =============================================================================
ENGINEERED_COL[0]#        'EF_RTP_of_birth'
new_df['EF_RTP_of_birth'] = dobs - unified_view['OC_MIN_timeperiodStart']

#dobs[0], unified_view['OC_MIN_timeperiodStart'][0]

ENGINEERED_COL[1] #        'EF_DIFF_EDD_DOB'
edd = unified_view['IN_EDD'].dropna().astype('datetime64[ns]')
edd = pd.to_datetime(unified_view['IN_EDD'].dropna())
new_df['EF_DIFF_EDD_DOB'] = dobs - edd

ENGINEERED_COL[2] #        'EF_DOW_DOB'
#valid_dates = unified_view[unified_view['OC_date_birth_vip']!='---']
#valid_dates = valid_dates['OC_date_birth_vip'].dropna().astype('datetime64[ns]').dt.strftime("%A")
#new_df['EF_DOW_DOB'] = valid_dates
#new_df['EF_DOW_DOB'].value_counts()
#
#ENGINEERED_COL[3] #        'EF_DOW_EDD'
#valid_dates = unified_view['IN_EDD'].dropna().astype('datetime64[ns]').dt.strftime("%A")
#new_df['EF_DOW_EDD'] = valid_dates
#new_df['EF_DOW_EDD'].value_counts()

#        'EF_Maternal_Age_Group'
x = unified_view[unified_view['IN_age']>0]
                   #0, 1, 2,  3,   4,    5 
age_bins = np.array([0, 13, 20, 30, 35, 99])
y = pd.Series(np.digitize(x['IN_age'],age_bins,right=False))
y.index = x.index
new_df['EF_Maternal_Age_Group'] = y
#new_df[['IN_age','EF_Maternal_Age_Group']]= y
#new_df['EF_Age_Under21']=0
#new_df['EF_Age_21to34']=0
#new_df['EF_Age_35andOver']=0
#new_df['EF_Age_Under21']=0
#new_df.loc[new_df['IN_age']<=20]['EF_Maternal_Age_Group']


#        'EF_Weeks_Preg_at_Presentation'
x = dobs - unified_view['OC_MIN_timeperiodStart']
x = x.fillna(0)
weeks = 40 - (x.dt.total_seconds() / (24 * 60 * 60)/7)
new_df['EF_Weeks_Preg_at_Presentation'] = weeks

#  column creation joining targets if at_home
new_df['EN_PLACE_BIRTH_at_home_AND_BLANKS'] = new_df['EN_PLACE_BIRTH_at_home']
new_df.loc[new_df['OC_place_birth']=="EN_PLACE_BIRTH_nan",'EN_PLACE_BIRTH_at_home_AND_BLANKS'] = 1
new_df.loc[new_df['OC_place_birth']=="EN_PLACE_BIRTH_---",'EN_PLACE_BIRTH_at_home_AND_BLANKS'] = 1

new_df.columns.values
# =============================================================================
# ENCODE TRUE, FALSE, NAN, ---
# =============================================================================
new_df.replace(['yes','no'],[1,0], inplace=True)
new_df.replace(True, 1, inplace=True)
new_df.replace(False, 0, inplace=True)
new_df.replace('---', 0, inplace=True)
new_df = new_df.fillna(0)
# questions for Neal
# - can we treat anc3 = blank/---/no as the same? (0)

# =============================================================================
# DYNAMIC COL COUNT
# =============================================================================
#case_ids = pd.Series(new_df.loc[:, UNIQUE_ID_COL_NAME].unique())
#a=3  
##TODO - ok we need to go through the list of cases first, then do the colums
## LOOP i - go through each case
#i =0
#for i in range(0, len(case_ids)): 
#    case = new_df[new_df['_id'] == case_ids[i]]
#    for a in range(0, len(DYNAMIC_COL_NAMES)):
#        match_cols = [col for col in new_df.columns if DYNAMIC_COL_NAMES[a]+'-RTP' in col]
#        match_cols.append('_id')
#        x = case[match_cols]
#        sumcol = 'EF_'+DYNAMIC_COL_NAMES[a]+'-sum'
#        asum = x.drop(UNIQUE_ID_COL_NAME, axis=1).values.sum()
#        new_df.loc[new_df['_id'] == case_ids[i], sumcol] = asum
 
         
##GET all the columns for this var
#for a in range(0, len(DYNAMIC_COL_NAMES)):
#    match_cols = [col for col in new_df.columns if DYNAMIC_COL_NAMES[a]+'-RTP' in col]
#    match_cols.append('_id')
#    x = new_df[match_cols]
#    i =0
#    sumcol = 'EF_'+DYNAMIC_COL_NAMES[a]+'-sum'
#    # LOOP i - go through each case
#    for i in range(0, len(case_ids)): 
#        asum = x.loc[x[UNIQUE_ID_COL_NAME] == case_ids[i]].drop(UNIQUE_ID_COL_NAME, axis=1).values.sum()
#        new_df.loc[new_df['_id'] == case_ids[i], sumcol] = asum
        


# =============================================================================
# OUTPUT
# =============================================================================
new_df = new_df.drop(['Unnamed: 0'], axis=1)
output_file_name = INPUT_FILE.rstrip('.csv') + '_EFC_2'+'.csv'
new_df.to_csv(output_file_name)
print(len(new_df.columns), " columns EF EN & C written to: ", output_file_name)


# =============================================================================
# TESTS
# =============================================================================
#Sanity stats on cleaned data
new_df.shape
new_df['OC_hamlet_name_vip'].value_counts()
new_df['Number pregnancy visits-RTP1'].value_counts()
new_df['EF_Weeks_Preg_at_Presentation'].describe()

unified_view.columns.values
new_df.columns.values

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
OUTPUT_DIR = config['DEFAULT']['OUTPUT_DIR']
UNIQUE_ID_COL_NAME = '_id'
STATIC_INPUT_COLS = ['IN_form.bp2.accompany_vip',
       'IN_form.has_bank_account_vip', 'IN_form.bp2.vehicle_vip',
       'IN_form.bp2.has_danger_signs_vip', 'IN_form.bp2.bleeding_vip',
       'IN_form.bp2.swelling_vip', 'IN_form.bp2.blurred_vision_vip',
       'IN_form.bp2.convulsions_vip', 'IN_form.bp2.rupture_vip'] #9

DYNAMIC_COL_NAMES = ['form.bp2.care_of_home_vip'] #1

OUTCOME_FACTOR_COLS = [
       'OC_form.where_born_vip', 'OC_form.delivery_nature_vip']

ENGINEERED_COL =['EF_tot_TP']

#ABS_FIRST_TP = datetime.strptime(str('01/01/2012'), '%m/%d/%Y') 

# =============================================================================
# Prep df and output df
# =============================================================================
unified_view = pd.read_csv(OUTPUT_DIR+INPUT_FILE)
new_df = unified_view

# =============================================================================
# CLEAN DATES
# =============================================================================
new_df['OC_MIN_timeperiodStart'] = pd.to_datetime(new_df['OC_MIN_timeperiodStart'].dropna())
new_df['OC_MIN_timeperiodStart'].value_counts().sort_index()
#dobs = pd.to_datetime(new_df[new_df['OC_date_birth_vip']!='---']['OC_date_birth_vip'].dropna())

# =============================================================================
# HOT ENCODING
# https://lukesingham.com/whos-going-to-leave-next/
# =============================================================================
for i in range(0, len(OUTCOME_FACTOR_COLS)): 
    new_df[OUTCOME_FACTOR_COLS[i]] = 'EN_'+OUTCOME_FACTOR_COLS[i].upper()[3:]+'_' + new_df[OUTCOME_FACTOR_COLS[i]].astype(str) 
    # generate dummies column encoding and join
    y = pd.get_dummies(new_df[OUTCOME_FACTOR_COLS[i]])  
    new_df = new_df.join(y)  

new_df.columns.values
# =============================================================================
# EF
# =============================================================================
#ENGINEERED_COL[0]#        'EF_RTP_of_birth'

#        'EF_Maternal_Age_Group'
#x = unified_view[unified_view['IN_age']>0]
#                   #0, 1, 2,  3,   4,    5 
#age_bins = np.array([0, 13, 20, 30, 35, 99])
#y = pd.Series(np.digitize(x['IN_age'],age_bins,right=False))
#y.index = x.index
#new_df['EF_Maternal_Age_Group'] = y

#        'EF_Weeks_Preg_at_Presentation'


#  column creation joining targets if at_home
new_df['EN_FORM.WHERE_BORN_VIP_home_AND_BLANKS'] = new_df['EN_FORM.WHERE_BORN_VIP_home']
new_df.loc[new_df['OC_form.where_born_vip']=="EN_FORM.WHERE_BORN_VIP_nan",'EN_FORM.WHERE_BORN_VIP_home_AND_BLANKS'] = 1
new_df['OC_form.where_born_vip'].value_counts()
new_df['EN_FORM.WHERE_BORN_VIP_home'].value_counts()
new_df['EN_FORM.WHERE_BORN_VIP_home_AND_BLANKS'].value_counts()
new_df.columns.values
# =============================================================================
# ENCODE TRUE, FALSE, NAN, ---
# =============================================================================
new_df.replace(['yes','no'],[1,0], inplace=True)
new_df.replace(True, 1, inplace=True)
new_df.replace(False, 0, inplace=True)
new_df.replace('---', 0, inplace=True)
new_df = new_df.fillna(0)          # might be key MaY, not sure if random forest can do blank fields
# questions for Neal

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
output_file_name = OUTPUT_DIR+INPUT_FILE.rstrip('.csv') + '_EFC_ABCDS'+'.csv'
new_df.to_csv(output_file_name)
print(len(new_df.columns), " columns EF EN & C written to: ", output_file_name)


# =============================================================================
# TESTS
# =============================================================================
#Sanity stats on cleaned data
new_df.shape
new_df.columns.values

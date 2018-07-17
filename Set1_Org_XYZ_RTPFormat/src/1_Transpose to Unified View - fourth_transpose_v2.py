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

import os as os
import datetime
import pandas as pd

#import my_functions as mf

START = datetime.datetime.now()
print('started:', START)

# PARAMATERS
#INPUT_FILE = 'T2_80r.csv'
#INPUT_FILE = 'preg_10K.csv'
#INPUT_FILE = 'preg_30k.csv'
#INPUT_FILE = 'preg cases period table trainSet 1 2018-02-06 - inc dob.csv'
#INPUT_FILE = 'TrainSet 1 20180206 w dob.csv'
#INPUT_FILE = 'TrainSet 2 20180310.csv'
#INPUT_FILE = 'T2_80r.csv'
INPUT_FILE = 'T2_80r_wrangler_cleaned.csv'

OUTPUT_DIR = 'Outputs P1/'
STATIC_INPUT_COLS = ['EDD','age', 'previous_pregnancies', 'living_children'] #4
DYNAMIC_COL_NAMES = ['openedInPeriod', 'closedInPeriod','Number_pregnancy_visits', 'anc_2', 'anc_3', 'anc_4', 'tetanus_1', 'takes_nutrition'] #7
STATIC_OUTCOME_COLS = [
        'anc_2_date', 'place_birth', 'type_delivery', 
        'maternal_outcome', 'pregnancy_outcome', 'birth_delivery_conducted_by', 'pre_term', 'birth_weight',
        'date_birth_vip','block_vip','hamlet_name_vip'
        ]#9
MAX_OUTCOME_COLS = []#['Months Til EDD']
MIN_OUTCOME_COLS = ['timeperiodStart', 'lmp_vip']
SUM_OUTCOME_COLS = ['Number_pregnancy_visits']
ENGINEERED_COL =['EF_tot_TP','EF_RTP_CLOSED']
UNIQUE_ID_COL_NAME = 'case'
LIMIT_TP = 10

# INITIATE
os.getcwd()
df = pd.read_csv(INPUT_FILE)                              #original dataframe

#transformed dataframe, initialised with the unique IDS
new_df = pd.DataFrame(df.loc[:, UNIQUE_ID_COL_NAME].unique())          
new_df.columns = ['_id']                                  #rename the column  
df.columns.values

# PREPARE_COLUMNS - pad DF with extra columns, so its human readable.   
# In the ORIGINAL df, it pulls the column of case_ids and counts how many instances there are in each (i.e. TP)
# num_TP = df[UNIQUE_ID_COL_NAME].value_counts()                                              
if LIMIT_TP > 0:
    max_tp = LIMIT_TP
else:
    max_tp = pd.DataFrame(df[UNIQUE_ID_COL_NAME].value_counts())['case'].max()                   

for AT_iterator in range(0, len(DYNAMIC_COL_NAMES)):
    for n in range(0, max_tp):
        var = DYNAMIC_COL_NAMES[AT_iterator]                           
        col = var +'-RTP'+ str(n+1)
        new_df.insert(len(new_df.columns), col, "")

#   Get a list of the unqiue case IDs
case_ids = pd.Series(df.loc[:, UNIQUE_ID_COL_NAME].unique())           
i=10
# LOOP i - go through each case
for i in range(0, len(case_ids)):                            
    #   Return only the rows for a case id
    current_case_rows = df.loc[df[UNIQUE_ID_COL_NAME] == case_ids[i]]   #returns a pd.DataFrame of all the rows for current case 
    current_case_rows = current_case_rows.sort_values('timeperiod', ascending=True)       #order them by timeperiod 
    current_case_rows = current_case_rows.reset_index()     #reset the indexes to 0,1,2...
    relativeTP = len(current_case_rows)                     #e.g. 7 for 'c1', 2 for 'c2'
    new_df.loc[new_df['_id'] == case_ids[i], 'EF_tot_TP'] = relativeTP
    
    #LOOP AT - go through each attribute
    for AT_iterator in range(0, len(DYNAMIC_COL_NAMES)):
        var = DYNAMIC_COL_NAMES[AT_iterator]                          #e.g. For ANC2 update each ANC val
        
        if LIMIT_TP > 0:
            if relativeTP > LIMIT_TP:
                relativeTP = LIMIT_TP
        #LOOP TP - go through each TimePeriod
        for TP_iterator in range(1, relativeTP+1):                         #iterates through all TP
            col = var +'-RTP'+ str(TP_iterator)
            new_df.loc[new_df['_id'] == case_ids[i], col] = current_case_rows.loc[TP_iterator-1, var] 
        #ADD DYNAMIC SUM COLUMN
        col = DYNAMIC_COL_NAMES[AT_iterator]+'-sum'
        new_df.loc[new_df['_id'] == case_ids[i], col] = current_case_rows[DYNAMIC_COL_NAMES[AT_iterator]].sum() 

    #LOOP STATIC INPUTS - copy over static attributes just as single colums
    #TODO: fragile to data quality as it assumes first date is the date
    for j in range(0,len(STATIC_INPUT_COLS)):
        new_df.loc[new_df['_id'] == case_ids[i], "IN_" + STATIC_INPUT_COLS[j]] = current_case_rows[STATIC_INPUT_COLS[j]].max()

    #LOOP STATIC OUTCOMES - copy over static attributes just as single colums
    for k in range(0, len(STATIC_OUTCOME_COLS)):
        if len(current_case_rows[STATIC_OUTCOME_COLS[k]].dropna()) == 1:
            new_df.loc[new_df['_id'] == case_ids[i], "OC_" + STATIC_OUTCOME_COLS[k]] = current_case_rows[STATIC_OUTCOME_COLS[k]].dropna().iloc[0]
        else:
            new_df.loc[new_df['_id'] == case_ids[i], "OC_" + STATIC_OUTCOME_COLS[k]] = current_case_rows.loc[0, STATIC_OUTCOME_COLS[k]]
    
    #LAST INSERTS FOR SPECIAL COLUMNS
    for l in range(0,len(MAX_OUTCOME_COLS)):
        new_df.loc[new_df['_id'] == case_ids[i], "OC_MAX_" + MAX_OUTCOME_COLS[l]] = current_case_rows[MAX_OUTCOME_COLS[l]].max()
        
    #LAST INSERTS FOR SPECIAL COLUMNS
    for m in range(0,len(MIN_OUTCOME_COLS)):
        if len(current_case_rows[MIN_OUTCOME_COLS[m]].dropna()) != 0:
            new_df.loc[new_df['_id'] == case_ids[i], "OC_MIN_" + MIN_OUTCOME_COLS[m]] = current_case_rows[MIN_OUTCOME_COLS[m]].dropna().iloc[0]
        else:
            new_df.loc[new_df['_id'] == case_ids[i], "OC_MIN_" + MIN_OUTCOME_COLS[m]] = current_case_rows.loc[0, MIN_OUTCOME_COLS[m]]
            
     #LAST INSERTS FOR SPECIAL COLUMNS
    for n in range(0,len(SUM_OUTCOME_COLS)):
        new_df.loc[new_df['_id'] == case_ids[i], "OC_SUM_" + SUM_OUTCOME_COLS[n]] = current_case_rows[SUM_OUTCOME_COLS[n]].sum()
   


# =============================================================================
# SAVE OUTPUT FILE 
# =============================================================================
print('Execution = ', datetime.datetime.now() - START)
output_file_name = OUTPUT_DIR+INPUT_FILE.rstrip('.csv') + '_output_'+ datetime.datetime.now().strftime("%Y%m%d-%Hh%M") +'.csv'
new_df.to_csv(output_file_name)
print(len(new_df), "rows, ", len(new_df.columns), " columns written to: ", output_file_name, ' saved in', os.getcwd())
del LIMIT_TP

# =============================================================================
# TESTS
# =============================================================================
df.shape        #58185 rows, 74 columns
new_df.shape    #9037 patients ?? columns
df.case.shape
df.EDD.describe()
df.EDD.dropna().shape
df.institutional_delivery_plan_vip.value_counts()   #8000 filled in
df.institutional_delivery_plan_vip.dropna().shape   #16331



# =============================================================================    


# =============================================================================
#  SPEED NOTES
# =============================================================================
# 10 input rows - instant
# 80 input rows, all TP - 1 sec
# 10k input rows, 3 TP - out 1516 rows,  21  columns - 1m47
# 10k input rows, 6 TP - out 1516 rows,  31  columns -  2m51 = +60%time for double columns
# 10k input rows, 10 TP - out 1516 rows,  63  columns -  3m19= +86%time for triple columns
# 10k input rows, 15 TP - out 1516 rows,  94  columns -  3m52= +117%time for quadruple columns
# 10k input rows, 25 TP - out 1516 rows,  154   columns -  19m36 = +1000d%time for quadruple columns
#
# 30k input rows, 3 TP - out 4647 rows,  22  columns - 6m40 =
# 30k input rows, 10 TP - out 4647 rows,  64  columns - 11m20 = +70% for triple columns
#
# 58k input rows, 10 TP - out 9016 rows, 64 columns - 40m21 = +505%  - PC slept
# 58k input rows, 7 Dynamic , 10 TP - out 9170 rows, 75 columns - 32m59 = +395% 
# 58k input rows, 7 Dynamic , 10 TP - out 9170 rows, 87 columns - 54m24 = +716%  
# =============================================================================
(54*60+24)/400-1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:46:31 2018

@author: ksharpey
"""
# =============================================================================
# This script 
#  - produces some simple stats on the unified view
#  - produces some simple Engineered Features
#  - produces some correlation insights
# =============================================================================

import os as os
import pandas as pd
import seaborn as sns

#constants
INPUT_FILE = 'preg_30k_output_20180209-10h41.csv'
#INPUT_FILE = 'preg_10k_output_20180208-11h09 - TP10.csv'
DYNAMIC_COL_NAMES = ['openedInPeriod-RTP1', 'openedInPeriod-RTP2',
       'openedInPeriod-RTP3', 'openedInPeriod-RTP4', 'openedInPeriod-RTP5',
       'openedInPeriod-RTP6', 'openedInPeriod-RTP7', 'openedInPeriod-RTP8',
       'openedInPeriod-RTP9', 'openedInPeriod-RTP10', 'closedInPeriod-RTP1',
       'closedInPeriod-RTP2', 'closedInPeriod-RTP3', 'closedInPeriod-RTP4',
       'closedInPeriod-RTP5', 'closedInPeriod-RTP6', 'closedInPeriod-RTP7',
       'closedInPeriod-RTP8', 'closedInPeriod-RTP9', 'closedInPeriod-RTP10',
       'anc_2-RTP1', 'anc_2-RTP2', 'anc_2-RTP3', 'anc_2-RTP4', 'anc_2-RTP5',
       'anc_2-RTP6', 'anc_2-RTP7', 'anc_2-RTP8', 'anc_2-RTP9', 'anc_2-RTP10',
       'anc_3-RTP1', 'anc_3-RTP2', 'anc_3-RTP3', 'anc_3-RTP4', 'anc_3-RTP5',
       'anc_3-RTP6', 'anc_3-RTP7', 'anc_3-RTP8', 'anc_3-RTP9', 'anc_3-RTP10',
       'anc_4-RTP1', 'anc_4-RTP2', 'anc_4-RTP3', 'anc_4-RTP4', 'anc_4-RTP5',
       'anc_4-RTP6', 'anc_4-RTP7', 'anc_4-RTP8', 'anc_4-RTP9', 'anc_4-RTP10',
       'tetanus_1-RTP1', 'tetanus_1-RTP2', 'tetanus_1-RTP3', 'tetanus_1-RTP4',
       'tetanus_1-RTP5', 'tetanus_1-RTP6', 'tetanus_1-RTP7', 'tetanus_1-RTP8',
       'tetanus_1-RTP9', 'tetanus_1-RTP10'] 
STATIC_COL_NAMES = ['OC_place_birth', 'OC_type_delivery']
ENGINEERED_COL =['tot_TP','tot_TP_before_birth','Max_TP_before_EDD',]
UNIQUE_ID_COL_NAME = '_id'

# INITIATE
os.getcwd()
unified_view = pd.read_csv(INPUT_FILE)
len(unified_view)
unified_view.columns

# =============================================================================
# FUNCTIONS
# =============================================================================
# =============================================================================
def value_count_pct(input_df, column_name=0):
    input_df = pd.DataFrame(input_df)
    x = pd.DataFrame(input_df[column_name].value_counts())
    x = x.rename(columns={ x.columns[0]: column_name })
    x.insert(len(x.columns),'pct',x[column_name]/len(input_df))
    #add blanks row
    blanks = pd.Series()
    blanks[column_name] = len(unified_view) - x[column_name].sum()
    blanks['pct'] = 1 - x['pct'].sum()
    blanks.name = 'blanks'
    x = x.append(blanks)
    x.sort_values(by=['pct']) 
    return(x)
     
# =============================================================================

# =============================================================================
# TESTS
# =============================================================================
# len(unified_view['_id']) == len(unified_view[UNIQUE_ID_COL_NAME].unique())
# =============================================================================


# =============================================================================
# STATS ZONE
# =============================================================================
# OUTCOMES
for AT_iterator in range(0,len(STATIC_COL_NAMES)):
        print(value_count_pct(unified_view,STATIC_COL_NAMES[AT_iterator]))
        print('')
# INPUTS
for AT_iterator in range(0,len(DYNAMIC_COL_NAMES)):
        print(value_count_pct(unified_view,DYNAMIC_COL_NAMES[AT_iterator]))
        print('')
        
x = value_count_pct(unified_view,STATIC_COL_NAMES[1])

# =============================================================================


# =============================================================================
# CORRELATIONS
# =============================================================================
x = sns.heatmap(UNIFIED_VIEW.corr(), annot=True, fmt=".2f")
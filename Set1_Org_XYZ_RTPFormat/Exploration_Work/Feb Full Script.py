#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:47:17 2018

@author: ksharpey
"""

import pandas as pd
import seaborn as sns

#df_full = pd.read_csv("preg cases period table trainSet 1 2018-02-04.csv")
df_full = pd.read_csv("preg cases period table trainSet 1 2018-02-06 - inc dob.csv")

# =============================================================================
# 'case', 'timeperiod', 'timeperiodStart', 'timeperiodEnd',
#        'openedInPeriod', 'closedInPeriod', 'Number pregnancy visits', 'anc_2',
#        'anc_2_date', 'anc_3', 'anc_3_date', 'anc_4', 'anc_4_date',
#        'tetanus_previous', 'tetanus_1', 'tt_1_date', 'tetanus_2', 'tt_2_date',
#        'IFA', 'takes_iron_folic', 'takes_nutrition', 'prepared_for_cost',
#        'institutional_delivery_plan', 'age_vip', 'previous_pregnancies_vip',
#        'living_children_vip', 'how_many_girl_child_vip',
#        'number_of_boy_child_vip', 'boy_children_vip', 'place_birth_vip',
#        'birth_weight_vip', 'type_delivery_vip',
#        'birth_delivery_conducted_by_vip', 'pre_term_vip',
#        'maternal_outcome_vip', 'pregnancy_outcome_vip',
#        'ANC2 Reported During Month', 'ANC3 Reported During Month',
#        'ANC4 Reported During Month', 'EDD', 'age', 'previous_pregnancies',
#        'living_children', 'how_many_girl_child', 'number_of_boy_child',
#        'boy_children', 'place_birth', 'type_delivery', 'maternal_outcome',
#        'pregnancy_outcome', 'birth_delivery_conducted_by', 'pre_term',
#        'birth_weight', 'Months Til EDD', 'Third Trimester', 'trainTestSet'],
# =============================================================================

# =============================================================================
# #STATS 0
# =============================================================================
pd.options.display.float_format = '{:.2f}%'.format
print('rows =',len(df_full['case']))                                        #58651
n_patients = len(df_full['case'].unique())
print('cases =',n_patients)                                                 #9170 avg 6 TP per case
n_opened = len(df_full.loc[df_full['openedInPeriod']==True])/n_patients
type(n_opened)
print('opened =',len(df_full.loc[df_full['openedInPeriod']==True]),' ', n_opened)         #8800 95% opened
print('closed =',len(df_full.loc[df_full['closedInPeriod']==True]), ' ', len(df_full.loc[df_full['closedInPeriod']==True])/n_patients)         #6995 77% closed (27% not closed)
print('birth outcomes =',len(df_full['pregnancy_outcome_vip'].dropna()), ' ', len(df_full['pregnancy_outcome_vip'].dropna())/n_patients)       

pd.reset_option('display.float_format')

# =============================================================================
# # STATS 1 understand the outcome targets
# =============================================================================
targets = [
        'date_birth_vip','anc_2_date', 'place_birth_vip', 'type_delivery_vip', 'maternal_outcome_vip', 'pregnancy_outcome_vip', 'birth_delivery_conducted_by_vip', 'pre_term_vip', 
        'birth_weight_vip',
        'pregnancy_outcome_vip']

pd.options.display.float_format = '{:.2f}%'.format
for i in range(0,len(targets)):
#    print("/n",targets[i]," ***************** ")
    a = pd.DataFrame(df_full[targets[i]].value_counts())
    a['percent'] = pd.DataFrame(df_full[targets[i]].value_counts(normalize=True))[targets[i]]*100
    print(a)
    print(df_full[targets[i]].value_counts().sum())
    
pd.reset_option('display.float_format')

# =============================================================================
#              place_birth_vip  percent
# in_hospital             5241   68.49%
# ---                     1234   16.13%
# at_home                 1144   14.95%
# in_transit                33    0.43%
# 7652
#            type_delivery_vip  percent
# normal                  5492   71.71%
# ---                     1901   24.82%
# caeserian                266    3.47%
# 7659
#                            maternal_outcome_vip  percent
# maternal_alive                             7013   91.40%
# ---                                         619    8.07%
# maternal_dead                                25    0.33%
# maternal_loss_to_followup                    16    0.21%
# 7673
#                       pregnancy_outcome_vip  percent
# live_birth                             6833   89.05%
# still_birth                             300    3.91%
# ---                                     255    3.32%
# spontaneous_abortion                    166    2.16%
# termination                             119    1.55%
# 7673
#                  birth_delivery_conducted_by_vip  percent
# anm                                         2231   29.14%
# dai_or_relative                             2117   27.65%
# nurse                                       2069   27.02%
# ---                                          840   10.97%
# doctor                                       399    5.21%
# 7656
#      pre_term_vip  percent
# no           6091   79.51%
# ---          1234   16.11%
# yes           336    4.39%
# 7661
#                birth_weight_vip  percent
# normal_weight              6034   78.80%
# ---                        1234   16.12%
# low_weight                  389    5.08%
# 7657
#                       pregnancy_outcome_vip  percent
# live_birth                             6833   89.05%
# still_birth                             300    3.91%
# ---                                     255    3.32%
# spontaneous_abortion                    166    2.16%
# termination                             119    1.55%
# 7673
# =============================================================================


# =============================================================================
# # STATS 3 understand targeted data
# =============================================================================
len(df_full.columns)            #57 columns
len(df_full)                    #58651 rows
len(df_full['case'].unique())   #9170 unique patient IDs
df_full.describe()
df_full['case'].value_counts().describe()
x = df_full['case'].value_counts()
x = x[x<=21]
x.name = 'timeperiods per case'
sns.distplot(x)
x.describe()
y = x[x<=13]
y.describe()
# =============================================================================
# count    9170.000000
# mean        6.395965
# std         5.481816
# min         1.000000
# 25%         3.000000
# 50%         5.000000
# 75%         7.000000
# max        60.000000
# =============================================================================
a = pd.DataFrame(df_full['case'].value_counts().value_counts()) 
a.sort_index() #9016
a['pct'] = a['case']/a['case'].sum()

# =============================================================================
#     case       pct
# 6   1199  0.130752
# 5   1150  0.125409
# 7   1029  0.112214
# 4    985  0.107415
# 2    942  0.102726
# 3    904  0.098582
# 1    773  0.084297
# 8    695  0.075791
# 9    368  0.040131
# 10   112  0.012214
# 11   102  0.011123
# 18    89  0.009706
# 12    83  0.009051
# 19    79  0.008615
# 20    67  0.007306
# =============================================================================

#EDD

edd = pd.DataFrame(df_full['Months Til EDD'].value_counts().sort_index())
edd.describe()
edd['pct'] = edd['Months Til EDD']/edd['Months Til EDD'].sum()
edd['edd']=edd.index
edd = edd[edd['pct']>0.001]
#edd['Months Til EDD'].name = 'Months Til EDD'
sns.distplot(df_full['Months Til EDD'].dropna())

#So, we need to figure out the max EDD per case
x = df_full.groupby('case')['Months Til EDD'].max()
#x = x[x>-2]
print(x.dropna().count(), 'of', df_full.case.nunique(), 'have EDD windows between', x.max(), 'and', x.min())
x.name = 'max(Months Til EDD)'
sns.distplot(x.dropna())
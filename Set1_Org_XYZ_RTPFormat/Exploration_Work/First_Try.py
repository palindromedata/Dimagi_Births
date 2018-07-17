#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:30:29 2018

@author: ksharpey
"""
# a script for fisrt pass of Neal's simpler pregnancy data
# using http://datascience-enthusiast.com/R/pandas_datatable.html
import pandas as pd
import os as os
import seaborn as sns

clear
os.getcwd()
df = pd.read_csv('Flat_File.csv')
# =============================================================================
# # col headings: 'case', 'timeperiod', 'timeperiodStart', 'timeperiodEnd',
#        'openedInPeriod', 'closedInPeriod', 'Number pregnancy visits', 'anc_2',
#        'anc_2_date', 'anc_3', 'anc_3_date', 'anc_4', 'anc_4_date',
#        'tetanus_previous', 'tetanus_1', 'tt_1_date', 'tetanus_2', 'tt_2_date',
#        'IFA', 'takes_iron_folic', 'takes_nutrition', 'prepared_for_cost',
#        'institutional_delivery_plan', 'ANC2 Reported During Month',
#        'ANC2 Reported During Or Before Month', 'ANC3 Reported During Month',
#        'ANC3 Reported During Or Before Month', 'ANC4 Reported During Month',
#        'ANC4 Reported During Or Before Month',
#        'Institutional Delivery Plan Reported During Or Before Month',
#        'Tetanus 1 Reported During Or Before Month',
#        'Tetanus 2 Reported During Or Before Month', 'EDD', 'place_birth',
#        'type_delivery', 'birth_delivery_conducted_by', 'pre_term',
#        'birth_weight', 'Months Til EDD', 'Third Trimester'
# =============================================================================

#simple stats
type(df)                #type
list(df)                #col names
len(df.index);          #number of rows
len(df.columns)         #num cols
df.shape                #dimension
df.describe()
df.head()

#investigation into some columns
df["birth_weight"]
df["birth_weight"].unique()
df["birth_weight"].value_counts()
x = df["birth_weight"].value_counts()
type(x)

df["type_delivery"].value_counts()
df["birth_delivery_conducted_by"].value_counts()
 len(df) #1675
 41/1675 #2% doctor!
df["type_delivery"].value_counts()
nvisits = df["case"].value_counts()
type(nvisits)
nvisits.head()
vstat = pd.DataFrame(nvisits.value_counts())
vstat.columns = ['cases']
vstat['pct']  = pd.Series(vstat['cases']/437*100)
vstat['cases'].sum()

vstat['cum']=0

for n in range(1,len(vstat.index)+1):
    if (n == 1):
        vstat.loc[n,'cum'] = vstat.loc[n,'pct']
    else:
        vstat.loc[n,'cum']= vstat.loc[n-1,'cum']+vstat.loc[n,'pct']
    print(vstat.loc[n,'pct'], vstat.loc[n,'cum'])

#group by one row
cases = df["case"].unique()

x = df.groupby(by='case')

#basic vis gaps
sns.pairplot(df, hue='case', vars='place_birth')#,'Number pregnancy visits','birth_delivery_conducted_by', 'pre_term','birth_weight', 'Months Til EDD'])



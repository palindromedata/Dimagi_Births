#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 02:05:02 2018

@author: ksharpey
"""

# Delivery table ABCDS, wider from with added columns for presentation date and last known visit

import os as os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from sklearn import metrics 
from sklearn.metrics import accuracy_score
from  sklearn.metrics import roc_auc_score

import datetime
START = datetime.datetime.now()

# =============================================================================
# CONFIG LOAD/ SETTINGS
# =============================================================================
os.getcwd()
df = pd.read_csv('Cleaned delivery table ABCDS 2018-06-26.csv')
#df['form_safe'].value_counts(normalize=True)
# What is form_safe? - NL: form_safe: 1= no deaths, 0= someone died
TARGET = 'caesarean'#'home'  #
#SINGLE_BEST = 'totalTime_20daysPriorToBirth'
#dropcols =['form_add', 'form_edd', 'del_completed_time', 'form_safe']
dropcols =['form_add', 'form_edd', 'del_completed_time', 'form_safe','caseid','firstVisitBeforeDelivery','lastVisitBeforeDelivery']
drop_targets =['form_where_born', #'totalTime_20daysPriorToBirth',
               'home', 
               'hospital', 
               'transit',
       'form_delivery_nature', 
#       'caesarean', 
       'instrumental', 'Empty',
       'unknown', 
       'vaginal'
       ]
##      HELPERS
print("shape = ", df.shape) 
df.columns.values
SHOW_GRAPHS = True
DATA_BALANCING = True
pd.set_option('display.width', 1000)
# =============================================================================
# DATE CLEANING & EF
# =============================================================================
temp_dates = pd.DataFrame()
temp_dates['form_add'] = pd.to_datetime(df['form_add'], format="%d/%m/%Y")
temp_dates['form_edd'] = pd.to_datetime(df['form_edd'], format="%d/%m/%Y")
temp_dates['form_edd'].describe()
temp_dates['add_edd_diff'] = temp_dates['form_edd'] - temp_dates['form_add'] 
temp_dates['days_diff'] = (temp_dates['add_edd_diff']/np.timedelta64(1, 'D')).astype(int)
temp_dates.head()
temp_dates['days_diff'].describe()
#df['days_diff'] = temp_dates['days_diff']

if pd.Series(['firstVisitBeforeDelivery','lastVisitBeforeDelivery']).isin(df.columns).all():
    temp_dates['firstVisitBeforeDelivery'] = pd.to_datetime(df['firstVisitBeforeDelivery'], format="%d/%m/%Y %H:%M")
    temp_dates['lastVisitBeforeDelivery'] = pd.to_datetime(df['lastVisitBeforeDelivery'], format="%d/%m/%Y %H:%M")
    #Get deltas to EDD
    days_diff = temp_dates['form_edd'] - temp_dates['firstVisitBeforeDelivery']
    temp_dates['1stV_days_to_EDD'] = (days_diff/np.timedelta64(1, 'D')).astype(int)
    days_diff = temp_dates['form_edd'] - temp_dates['lastVisitBeforeDelivery']
    temp_dates['LastV_days_to_EDD'] = (days_diff/np.timedelta64(1, 'D')).astype(int)
    df['1stV_days_to_EDD'] = temp_dates['1stV_days_to_EDD']
#    df['LastV_days_to_EDD'] = temp_dates['LastV_days_to_EDD']
    
temp_dates.head()

#df[['form_add','form_edd','days_diff']].head()
## SEASON OF EDD, SEASON OF INITIATION, SEASON OF EDD

# =============================================================================
# TIME CLEAN
# =============================================================================
#df['totalTime_Hours'] = (df['totalTime_20daysPriorToBirth']/60).astype(int) #*60
#df['totalTime_Hours'].describe()
#df['totalTime_Hours'].value_counts()
# =============================================================================
# INITIAL STATS PREVELANCE
# =============================================================================
#df['form_where_born'].value_counts(normalize=True)
#df['form_delivery_nature'].value_counts(normalize=True)
#df['totalTime_20daysPriorToBirth'].value_counts(normalize=True)
#df['totalTime_20daysPriorToBirth'].describe()
##check cols
#for col in df.columns.values:
#    print (col, " : ", df[col].iloc[0], type(df[col].iloc[0]))

# =============================================================================
# MANUAL BALANCING
# =============================================================================
# should find the minority_set and rebalance
df.shape
df[TARGET].value_counts(normalize = True)
true_set = df[df[TARGET]==1]
true_set.shape
true_set[TARGET].value_counts(normalize = True)
false_set = df[df[TARGET]==0]
false_set.shape
false_set[TARGET].value_counts(normalize = True)
if (true_set.shape[0]>false_set.shape[0]):
    big_set = true_set
    small_set = false_set
else:
    big_set = false_set
    small_set = true_set
    
sample = big_set.sample(n=small_set.shape[0])
sample.shape
sample[TARGET].value_counts(normalize = True)
final = pd.concat([small_set,sample])
final.shape
final[TARGET].value_counts(normalize = True)

if DATA_BALANCING:
    df = final

# =============================================================================
# DROP DATE OR TEXT COLUMNS
# =============================================================================
df.drop(dropcols,axis=1,inplace=True)
df.drop(drop_targets,axis=1,inplace=True)
print("cleaned shape = ", df.shape)
# =============================================================================
# SPLIT & TRAIN
# =============================================================================
X = df
#X = df[[TARGET, SINGLE_BEST]]
Y = df[TARGET]
X.drop(TARGET, axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.4, random_state=42)
#model = LogisticRegression()
#model = RandomForestClassifier() 
model = GradientBoostingClassifier()
#model = XGBClassifier()
model.fit(x_train, y_train) 

# =============================================================================
# TESTING AND RESULTS
# =============================================================================
# Predictions/probs on the test dataset
predicted = pd.DataFrame(model.predict(x_test))  
probs = pd.DataFrame(model.predict_proba(x_test))

# Store metrics
model_accuracy = metrics.accuracy_score(y_test, predicted)  
model_roc_auc = metrics.roc_auc_score(y_test, probs[1])  
model_confus_matrix = metrics.confusion_matrix(y_test, predicted)  
model_classification_report = metrics.classification_report(y_test, predicted)  
model_precision = metrics.precision_score(y_test, predicted, pos_label=1)  
model_recall = metrics.recall_score(y_test, predicted, pos_label=1)  
model_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print("target = ", TARGET)
print(type(model), "\naccuracy: ",model_accuracy, ", model_roc_auc: ", model_roc_auc)
print("confusion matrix \n", model_confus_matrix)
print("prevelence in y_test:")
print(y_test.value_counts(normalize = True))
print(y_test.value_counts( ))

# LITTLE TEST TO BUCKET THE RISK INTO human readable buckets #.# - doesn't seem to work well at all impact the ranking test
#probs = pd.DataFrame(model.predict_proba(x_test))
#probs[1] = round(probs[1],2)
#probs[1].value_counts(normalize=True)


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print ("\nFeatures sorted by their score: ", "target = ", TARGET)
mylist =pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), x_train), reverse=True))
mylist.columns = ['feature_importance','feature']
mylist['cum_importance']= mylist['feature_importance'].cumsum()
print(mylist[0:10])


# =============================================================================
# OPTIMAL FEATURES
# =============================================================================
# =============================================================================
# from sklearn.feature_selection import RFECV
# import matplotlib.pyplot as plt
# # The "accuracy" scoring is proportional to the number of correct classifications
# rfecv = RFECV(estimator=model, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
# rfecv = rfecv.fit(x_train, y_train)
# print('Optimal number of features :', rfecv.n_features_)
# print('Best features :', x_train.columns[rfecv.support_])
## 
## # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score of number of selected features")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()
# =============================================================================

# =============================================================================
# PCT PRIORITISATION
# =============================================================================
p_test = df
p_test.shape
#p_test.drop(TARGET, axis=1, inplace=True)
p_test['probs'] = pd.DataFrame(model.predict_proba(p_test))[1]
p_test['index1'] = p_test.index
p_test.columns.values
cols = ['anyYes20daysPriorToBirth_form_has_bank_account', 'probs']
col = 'index1'
SHOW_GRAPHS = True
p_test[TARGET] = Y
#iterate through
for col in cols:
    #prepcumsum
    x = pd.DataFrame(p_test.sort_values(col,ascending=True)[[col,TARGET]]) 
    x['count'] = range(1,len(x)+1)
    x['target_cumsum'] = x[TARGET].cumsum()
    x["pct by count"] = x["count"]/len(x) 
    x["pct_by_cumsum"] = x["target_cumsum"]/x[TARGET].sum() 
    if SHOW_GRAPHS:
        x.plot(x="pct by count", y="pct_by_cumsum", marker='.')
    percentiles = [0.1,0.5,1,5,10,20,30,50,75,90]
    for p in percentiles:
        print("ORDERED BY: ", col,", range = ", p, "percentile, TARGET = ",TARGET )
        y = x.loc[x["count"]==round(len(x)*p/100),:] 
        lift = y.iloc[0,len(y.columns)-1]-y.iloc[0,len(y.columns)-2]
        multiple = y.iloc[0,len(y.columns)-1]/y.iloc[0,len(y.columns)-2]
        print(y,"lift=","{:.00%}".format(lift), "or ", "%.1fx" % multiple)

temp = x.head(n=5000)
x.head(n=50)
temp[TARGET].value_counts()
x[TARGET].value_counts(normalize=True)
# =============================================================================
# END OF FILE
# =============================================================================
print('Execution = ', datetime.datetime.now() - START)
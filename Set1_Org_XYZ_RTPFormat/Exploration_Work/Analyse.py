#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:44:45 2018

@author: ksharpey
"""
# =============================================================================
# Description
#  - simple correlation
#  - pair plot
#  - binary classifier
#  
#  does not run ML
# =============================================================================

import os as os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.cross_validation import cross_val_score  
from sklearn.metrics import roc_curve
from sklearn.dummy import DummyClassifier
from lifelines import KaplanMeierFitter
from ggplot import *


#constants
OUTPUT_DIR = 'Outputs/'
#INPUT_FILE = OUTPUT_DIR +'T2_80r_output_20180313-11h09_EFC.csv'
INPUT_FILE = OUTPUT_DIR +'TrainSet 2 20180310_output_20180313-14h56_EFC_hand encode.csv'
#INPUT_FILE = OUTPUT_DIR +'preg cases period table trainSet 1 2018-02-06 - inc dob_output_20180226-16h06_EFC.csv'
#INPUT_FILE = OUTPUT_DIR +'preg cases period table trainSet 1 2018-02-06 - inc dob_output_20180226-16h06_EFC.csv'
TEST_FILE = OUTPUT_DIR +'preg cases period table trainSet 1 2018-02-06 - inc dob_output_20180226-16h06_EFC - days clean.csv'


# INITIATE
os.getcwd()
unified_view = pd.read_csv(INPUT_FILE)
filtered_view = unified_view
test_set = pd.read_csv(TEST_FILE) 

# =============================================================================
# REMOVE MASSIVE OUTLIERS
# =============================================================================
valid_EDD = unified_view[unified_view['IN_EDD']!='01/01/1970'] #3926 T1, T2 
valid_EDD = valid_EDD[valid_EDD['IN_age']!=0]               #3905 T1, T2 3850
filtered_view = valid_EDD

x = filtered_view.loc[:, [
#x = unified_view.loc[:, [
       '_id','IN_age', 'IN_previous_pregnancies', 'IN_living_children',
       'OC_MAX_Months Til EDD', 'OC_SUM_Number pregnancy visits', 'EF_tot_TP',
       'EF_Maternal_Age_Group','EF_Weeks_Preg_at_Presentation',
       'EN_PLACE_BIRTH_in_hospital','EN_BIRTH_WEIGHT_low_weight','EN_PREGNANCY_OUTCOME_live_birth','EN_PRE_TERM_yes',
#       'EF_openedInPeriod-sum', 'EF_closedInPeriod-sum',  'EF_anc_2-sum',  'EF_anc_3-sum',  'EF_anc_4-sum',  'EF_tetanus_1-sum',  'EF_takes_nutrition-sum'
       'OC_block_vip_h','OC_hamlet_name_vip_h'
       ]]
x.shape

# =============================================================================
# CORRELATIONS
# =============================================================================

#grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
#f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
plt.figure(figsize = (16,5))
out = sns.heatmap(x.corr(), annot=True, fmt=".1f")

out = sns.heatmap(filtered_view.corr(), annot=True, fmt=".2f") 

# =============================================================================
# PAIRPLOT - almost a write off
# =============================================================================
y = x
pplot = sns.pairplot(y, hue="OC_birth_weight")  
pplot = sns.pairplot(y, hue='OC_pre_term')  
pplot = sns.pairplot(y, hue='EN_PREGNANCY_OUTCOME_live_birth') 
pplot = sns.pairplot(y, hue='EN_PLACE_BIRTH_in_hospital') 


# =============================================================================
# Joint plot correlation
# =============================================================================
sns.jointplot("IN_age", "IN_previous_pregnancies", data=x, kind="reg")
sns.jointplot("IN_age", "IN_previous_pregnancies", data=x, kind="reg", stat_func=spearmanr)
sns.jointplot("EF_Weeks_Preg_at_Presentation", "EN_PLACE_BIRTH_in_hospital", data=x, kind="reg")
sns.jointplot("EF_Weeks_Preg_at_Presentation", "EN_PLACE_BIRTH_in_hospital", data=x, kind="reg", stat_func=spearmanr)


# =============================================================================
# BINARY CLASSIFIERS ---- PREPARATION
# https://lukesingham.com/whos-going-to-leave-next/
# =============================================================================
#       PREPARATION
target = 'EN_PLACE_BIRTH_in_hospital'
df = filtered_view

# remove tricky date & text columns
dropcols = ['EF_RTP_of_birth','EF_DIFF_EDD_DOB', 'IN_EDD', 
       'OC_date_birth_vip', 'OC_anc_2_date','OC_MIN_timeperiodStart',
       'OC_MIN_lmp_vip']
# remove tricky date & text columns
dropcols.extend(['Unnamed: 0','_id','OC_place_birth', 'OC_type_delivery', 'OC_maternal_outcome',
       'OC_pregnancy_outcome', 'OC_birth_delivery_conducted_by',
       'OC_pre_term', 'OC_birth_weight'])
#TODO these need to be encoded
dropcols.extend(['OC_block_vip','OC_hamlet_name_vip'])
# remove predictors
dropcols.extend([
#         'EN_PLACE_BIRTH_in_hospital',   
         'EN_PLACE_BIRTH_at_home',   
         'EN_BIRTH_WEIGHT_normal_weight',
         'EN_PRE_TERM_no', 
         'EN_PRE_TERM_yes',
       'EN_PLACE_BIRTH_in_transit', 'EN_PLACE_BIRTH_nan',
       'EN_TYPE_DELIVERY_---', 'EN_TYPE_DELIVERY_caeserian',
       'EN_TYPE_DELIVERY_nan', 'EN_TYPE_DELIVERY_normal',
       'EN_MATERNAL_OUTCOME_---', 'EN_MATERNAL_OUTCOME_maternal_alive',
       'EN_MATERNAL_OUTCOME_maternal_dead',
       'EN_MATERNAL_OUTCOME_maternal_loss_to_followup',
       'EN_MATERNAL_OUTCOME_nan', 'EN_PREGNANCY_OUTCOME_---',
       'EN_PREGNANCY_OUTCOME_live_birth', 'EN_PREGNANCY_OUTCOME_nan',
       'EN_PREGNANCY_OUTCOME_spontaneous_abortion',
       'EN_PREGNANCY_OUTCOME_still_birth',
       'EN_PREGNANCY_OUTCOME_termination',
       'EN_BIRTH_DELIVERY_CONDUCTED_BY_---',
       'EN_BIRTH_DELIVERY_CONDUCTED_BY_anm',
       'EN_BIRTH_DELIVERY_CONDUCTED_BY_dai_or_relative',
       'EN_BIRTH_DELIVERY_CONDUCTED_BY_doctor',
       'EN_BIRTH_DELIVERY_CONDUCTED_BY_nan',
       'EN_BIRTH_DELIVERY_CONDUCTED_BY_nurse', 'EN_PRE_TERM_---',
       'EN_PRE_TERM_nan', 
       'EN_BIRTH_WEIGHT_---', 'EN_BIRTH_WEIGHT_low_weight',
       'EN_BIRTH_WEIGHT_nan','EN_PLACE_BIRTH_---',
       'EF_tot_TP','OC_SUM_Number pregnancy visits', 'OC_MAX_Months Til EDD',
       'openedInPeriod-sum', 'closedInPeriod-sum',  'anc_2-sum',  'anc_3-sum',  'anc_4-sum',  'tetanus_1-sum',  'takes_nutrition-sum', 'Number pregnancy visits-sum'
#        'EF_openedInPeriod-sum', 'EF_closedInPeriod-sum',  'EF_anc_2-sum',  'EF_anc_3-sum',  'EF_anc_4-sum',  'EF_tetanus_1-sum',  'EF_takes_nutrition-sum'
         ])
df.columns.values
len(dropcols)
df = df.drop(dropcols,axis=1)
df.shape
# =============================================================================
# # remove tricky date & text columns
# df = df.drop(
#         ['EF_RTP_of_birth','EF_DIFF_EDD_DOB', 'IN_EDD', 
#        'OC_date_birth_vip', 'OC_anc_2_date','OC_MIN_timeperiodStart',
#        'OC_MIN_lmp_vip'],axis=1)
# # remove tricky date & text columns
# df = df.drop(
#         ['Unnamed: 0','_id','OC_place_birth', 'OC_type_delivery', 'OC_maternal_outcome',
#        'OC_pregnancy_outcome', 'OC_birth_delivery_conducted_by',
#        'OC_pre_term', 'OC_birth_weight'],axis=1)
# #TODO these need to be encoded
# df = df.drop(
#         ['OC_block_vip','OC_hamlet_name_vip'
#          ],axis=1)
# # remove predictors  
# df = df.drop(
#         [
# #         'EN_PLACE_BIRTH_in_hospital',   
#          'EN_PLACE_BIRTH_at_home',   
#          'EN_BIRTH_WEIGHT_normal_weight',
#          'EN_PRE_TERM_no', 
#          'EN_PRE_TERM_yes',
#        'EN_PLACE_BIRTH_in_transit', 'EN_PLACE_BIRTH_nan',
#        'EN_TYPE_DELIVERY_---', 'EN_TYPE_DELIVERY_caeserian',
#        'EN_TYPE_DELIVERY_nan', 'EN_TYPE_DELIVERY_normal',
#        'EN_MATERNAL_OUTCOME_---', 'EN_MATERNAL_OUTCOME_maternal_alive',
#        'EN_MATERNAL_OUTCOME_maternal_dead',
#        'EN_MATERNAL_OUTCOME_maternal_loss_to_followup',
#        'EN_MATERNAL_OUTCOME_nan', 'EN_PREGNANCY_OUTCOME_---',
#        'EN_PREGNANCY_OUTCOME_live_birth', 'EN_PREGNANCY_OUTCOME_nan',
#        'EN_PREGNANCY_OUTCOME_spontaneous_abortion',
#        'EN_PREGNANCY_OUTCOME_still_birth',
#        'EN_PREGNANCY_OUTCOME_termination',
#        'EN_BIRTH_DELIVERY_CONDUCTED_BY_---',
#        'EN_BIRTH_DELIVERY_CONDUCTED_BY_anm',
#        'EN_BIRTH_DELIVERY_CONDUCTED_BY_dai_or_relative',
#        'EN_BIRTH_DELIVERY_CONDUCTED_BY_doctor',
#        'EN_BIRTH_DELIVERY_CONDUCTED_BY_nan',
#        'EN_BIRTH_DELIVERY_CONDUCTED_BY_nurse', 'EN_PRE_TERM_---',
#        'EN_PRE_TERM_nan', 
#        'EN_BIRTH_WEIGHT_---', 'EN_BIRTH_WEIGHT_low_weight',
#        'EN_BIRTH_WEIGHT_nan','EN_PLACE_BIRTH_---',
#        'EF_tot_TP','OC_SUM_Number pregnancy visits', 'OC_MAX_Months Til EDD',
#        'openedInPeriod-sum', 'closedInPeriod-sum',  'anc_2-sum',  'anc_3-sum',  'anc_4-sum',  'tetanus_1-sum',  'takes_nutrition-sum', 'Number pregnancy visits-sum'
# #        'EF_openedInPeriod-sum', 'EF_closedInPeriod-sum',  'EF_anc_2-sum',  'EF_anc_3-sum',  'EF_anc_4-sum',  'EF_tetanus_1-sum',  'EF_takes_nutrition-sum'
#          ],axis=1)
# 
# =============================================================================
#       SPLIT
# Randomly, split the data into test/training/validation sets
train, test, validate = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])  
print(train.shape, test.shape, validate.shape)  #  
# =============================================================================
# # OR use another file for test
# =============================================================================
train = df
test,validate = np.split(test_set.sample(frac=1), 2) 

mdrop = ['OC_MIN_lmp_vip' 'OC_block_vip' 'OC_hamlet_name_vip' 'openedInPeriod-sum'
 'closedInPeriod-sum' 'anc_2-sum' 'anc_3-sum' 'anc_4-sum' 'tetanus_1-sum'
 'takes_nutrition-sum' 'Number pregnancy visits-sum']
type(mdrop)
b = [a for a in dropcols if a not in mdrop]
len(b)

test = test.drop(dropcols,axis=1) 
print(train.shape, test.shape, validate.shape)  # 
# =============================================================================

filtered_view.shape
df.shape
# Separate target and predictors
y_train = train[target]  
x_train = train.drop([target], axis=1)  
y_test = test[target]  
x_test = test.drop([target], axis=1)  
y_validate = validate[target]  
x_validate = validate.drop([target], axis=1)

# Check the balance of the splits on y_
y_test.mean() # 0.5505
y_train.mean() # 0.5556

#       TRAIN
rf = RandomForestClassifier()  
rf.fit(x_train, y_train) 

#       VAR IMPORTANCE TEST
 
print ("Features sorted by their score:"  )
mylist = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True) 
mylist = pd.DataFrame(mylist)
mylist.columns = ['feature_importance','feature']
mylist['cum_importance']= mylist['feature_importance'].cumsum()
mylist.plot()

# Create variable lists and drop
all_vars = x_train.columns.tolist()  
top_5_vars =['EF_Weeks_Preg_at_Presentation', 'IN_age',
        'IN_previous_pregnancies']  
bottom_vars = [cols for cols in all_vars if cols not in top_5_vars]
# Drop less important variables leaving the top_5
x_train    = x_train.drop(bottom_vars, axis=1)  
x_test     = x_test.drop(bottom_vars, axis=1)  
x_validate = x_validate.drop(bottom_vars, axis=1)  


# =============================================================================
# REGRESSION - http://blog.yhat.com/posts/roc-curves.html
# =============================================================================
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
logistic_model.score(x_train,y_train)

#check prediction performance
predicted = pd.DataFrame(logistic_model.predict(x_test))
probs = pd.DataFrame(logistic_model.predict_proba(x_test))
metrics.accuracy_score(y_test,predicted)
metrics.confusion_matrix(y_test,predicted)
metrics.classification_report(y_test,predicted)

# Build the ROC
preds = logistic_model.predict_proba(x_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
#calc AUC
auc = metrics.auc(fpr,tpr)
ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +\
    geom_area(alpha=0.2) +\
    geom_line(aes(y='tpr')) +\
    ggtitle("ROC Curve w/ AUC=%s" % str(auc))

# =============================================================================
# BINARY CLASSIFIERS -- TESTS
# =============================================================================
rf_model = rf.fit(x_train, y_train)  
# training accuracy 98.71%
rf_model.score(x_train, y_train)
predicted = pd.DataFrame(rf_model.predict(x_test))  

# Predictions/probs on the test dataset
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# Store metrics
rf_accuracy = metrics.accuracy_score(y_test, predicted)  
rf_roc_auc = metrics.roc_auc_score(y_test, predicted)  # is there a bug here?
rf_confus_matrix = metrics.confusion_matrix(y_test, predicted)
rf_classification_report = metrics.classification_report(y_test, predicted)  
rf_precision = metrics.precision_score(y_test, predicted, pos_label=1)  
rf_recall = metrics.recall_score(y_test, predicted, pos_label=1)  
rf_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print(rf_classification_report)


# Evaluate the model using 10-fold cross-validation
rf_cv_scores = cross_val_score(RandomForestClassifier(), x_test, y_test, scoring='precision', cv=10)  
rf_cv_mean = np.mean(rf_cv_scores)  

# Model comparison
models = pd.DataFrame({  
  'Model': ['r.f.'],
  'Accuracy' : [rf_accuracy],
  'Precision': [rf_precision],
  'recall' : [ rf_recall],
  'F1' : [ rf_f1],
  'cv_precision' : [rf_cv_mean]
})
# Print table and sort by test precision
print('Target = ', target)
models.sort_values(by='Precision', ascending=False)


# =============================================================================
# ## DUMMY 
# =============================================================================
dummy = DummyClassifier()
dummy = DummyClassifier(strategy='uniform', random_state = 100, constant = None) 
dummy.fit(x_train, y_train)
dumm_pred = dummy.predict(x_test) 

#scoring
dumm_accuracy = metrics.accuracy_score(y_test, dumm_pred)  
dumm_roc_auc = metrics.roc_auc_score(y_test, dumm_pred) 
dumm_confus_matrix = metrics.confusion_matrix(y_test, dumm_pred)
dumm_classification_report = metrics.classification_report(y_test, dumm_pred)  
dumm_precision = metrics.precision_score(y_test, dumm_pred, pos_label=1)  
dumm_recall = metrics.recall_score(y_test, dumm_pred, pos_label=1)  
dumm_f1 = metrics.f1_score(y_test, dumm_pred, pos_label=1)

print(dumm_classification_report)
type(rf_classification_report)
# =============================================================================
    

# =============================================================================
# BASELINE
# =============================================================================
# BASELINE
unified_view.EN_PLACE_BIRTH_in_hospital.value_counts()
bases = pd.DataFrame({'All data UV': unified_view[target],"Filtered_View": filtered_view[target],"y_train": y_train, "y_test": y_test})
print("Baselines:")
for col in bases:
    print(bases[col].value_counts()/len(bases[col].dropna())*100, "n = ",len(bases[col].dropna()))



# =============================================================================
# SURVIVAL
# =============================================================================
## Note to self -> this might be better fit for the raw data

kmf = KaplanMeierFitter()
data = filtered_view
T = data['EF_Weeks_Preg_at_Presentation']
E = data['EN_PLACE_BIRTH_in_hospital']
kmf.fit(T, event_observed=E)
kmf.survival_function_.plot()
plt.title('hellow world');
kmf.plot()

# =============================================================================
# TESTS
# =============================================================================

unified_view.columns
match_cols = [col for col in unified_view.columns if '-sum' in col]
for i in range(0, len(match_cols)):
for x in match_cols:
    print(x)
    print(unified_view[x].describe())
    print(unified_view[x].value_counts())

filtered_view['EF_Weeks_Preg_at_Presentation'] = round(unified_view['EF_Weeks_Preg_at_Presentation']).astype(int) # 36.8, 23.0
qa = filtered_view[['OC_MAX_Months Til EDD','EF_Weeks_Preg_at_Presentation','IN_EDD','OC_MIN_timeperiodStart','OC_date_birth_vip']]   #.sum() =9170

qa.plot.hexbin('EF_Weeks_Preg_at_Presentation','OC_MAX_Months Til EDD')
qa.to_csv("temp_date_check.csv")
# Takeaways: 'OC_MAX_Months Til EDD' 
#   - needs to be forgiving up to -1. convert these to legit 0's 
#   - some 0's just mean born in same month as registered  -is ok
#   - should be renamed tochanged to weeks before EDD

# Takeaways: 'EF_Weeks_Preg_at_Presentation'
#   - rename 'EF_Presentation_weeks_prebirth'
#   - recalc: 

# Takaways: 'IN_age'
#    - 3905 with ages

# Takaways: 'IN_EDD'
#    - can be >10 weeks off DOB
#    - 3926 with EDD

# Takeaways: OC_MIN_timeperiodStart
#   - should be an IN
# =============================================================================
# FUNCTIONS
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
# STATS ZONE
# =============================================================================
        
for i in range(0,len(new_df.columns)):
        print(value_count_pct(new_df,new_df.columns[i]))
        print('')
        
# =============================================================================


# Create linear regression object
regr = linear_model.LinearRegression()
 
X = x['IN_age'].dropna()
X.reshape(len(X),1)
Y = x['IN_previous_pregnancies'].dropna()
Y.reshape(len(Y),1)
# Train the model using the training sets
plt.scatter(x, y,  color='black')
regr.fit(x['IN_age'], x['IN_previous_pregnancies'])
 
# Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)
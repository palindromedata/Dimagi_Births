#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:37:41 20185

@author: ksharpey
"""
import os as os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
from sklearn.cross_validation import cross_val_score  
from sklearn.metrics import roc_curve
from sklearn.dummy import DummyClassifier
from sklearn import model_selection
from sklearn.feature_selection import RFECV
from ggplot import *
import configparser
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import datetime
START = datetime.datetime.now()

config = configparser.ConfigParser()
config.read('config_P2.ini')

INPUT_FILE = config['PREDICT']['INPUT_FILE']
#INPUT_FILE = OUTPUT_DIR +'TrainSet 2 20180310_output_20180313-14h56_EFC_hand encode.csv'
TEST_FILE = config['PREDICT']['TEST_FILE']
TARGETS = config.get('PREDICT','TARGETS').split(',')
VERBOSE = config.getboolean('DEFAULT','VERBOSE')
SHOW_GRAPHS = config.getboolean('DEFAULT','SHOW_GRAPHS')  #False

# INITIATE
os.getcwd()
pd.set_option('display.width', 1000)
unified_view = pd.read_csv(INPUT_FILE)

unified_view.columns.values
discovery = ['OC_place_birth','OC_type_delivery','OC_maternal_outcome','OC_pregnancy_outcome','OC_birth_delivery_conducted_by','OC_pre_term','OC_birth_weight']
for i in discovery:
    print(unified_view[i].value_counts(normalize=True))
    
    
# TEMP BLANKS FILTERING
unified_view = unified_view[unified_view['OC_date_birth_vip']!="0"]
unified_view.shape
   
if (TEST_FILE != ''):
    test_set_original = pd.read_csv(TEST_FILE) 
#    test_set_original = test_set_original[test_set_original['OC_date_birth_vip']!="0"]
#FULL
# =============================================================================
# EN_PLACE_BIRTH_in_hospital    5264/9170 = 57%
# EN_PLACE_BIRTH_nan            1510
# EN_PLACE_BIRTH_---            1220
# EN_PLACE_BIRTH_at_home        1143/9170 = 12%
# EN_PLACE_BIRTH_in_transit       33
# =============================================================================
#BLANK VIP filtered
# =============================================================================
# EN_PLACE_BIRTH_in_hospital    5183/6754 = 76%
# EN_PLACE_BIRTH_at_home        1131/6754 = 16%
# EN_PLACE_BIRTH_---             386/6754 = 6%
# EN_PLACE_BIRTH_in_transit       33
# EN_PLACE_BIRTH_nan              21
# ============================================================================= 

# FILTER OUT TP>=6  TODO
dropcols = unified_view.filter(regex='RTP4|RTP5|RTP6|RTP7|RTP8|RTP9|RTP10').columns
unified_view = unified_view.drop(dropcols,axis=1)

# EFC column creation & # if at_home
unified_view['EN_PLACE_BIRTH_at_home_AND_BLANKS'] = unified_view['EN_PLACE_BIRTH_at_home']
unified_view.loc[unified_view['OC_place_birth']=="EN_PLACE_BIRTH_nan",'EN_PLACE_BIRTH_at_home_AND_BLANKS'] = 1
unified_view.loc[unified_view['OC_place_birth']=="EN_PLACE_BIRTH_---",'EN_PLACE_BIRTH_at_home_AND_BLANKS'] = 1
#unified_view['EN_PLACE_BIRTH_at_home_AND_BLANKS'].value_counts(normalize=True)
#unified_view['EN_PLACE_BIRTH_at_home'].value_counts(normalize=True)

# EFC column creation & low_weight and any not normal weight
unified_view['EN_BIRTH_WEIGHT_low_weight_AND_BLANKS'] = unified_view['EN_BIRTH_WEIGHT_low_weight']
unified_view.loc[unified_view['OC_birth_weight']=="EN_BIRTH_WEIGHT_nan",'EN_BIRTH_WEIGHT_low_weight_AND_BLANKS'] = 1
unified_view.loc[unified_view['OC_birth_weight']=="EN_BIRTH_WEIGHT_---",'EN_BIRTH_WEIGHT_low_weight_AND_BLANKS'] = 1
#unified_view['EN_BIRTH_WEIGHT_low_weight_AND_BLANKS'].value_counts(normalize=True)
#unified_view['EN_BIRTH_WEIGHT_low_weight'].value_counts(normalize=True)

# =============================================================================
# BINARY CLASSIFIERS ---- PREPARATION
# https://lukesingham.com/whos-going-to-leave-next/
# =============================================================================
#       PREPARATION
df = unified_view

# remove tricky date & text columns
dropcols = ['EF_RTP_of_birth','EF_DIFF_EDD_DOB', 'IN_EDD', 
       'OC_date_birth_vip', 'OC_anc_2_date','OC_MIN_timeperiodStart',
#       'OC_MIN_lmp_vip'
        ]
# remove tricky date & text columns
dropcols.extend(['_id','OC_place_birth', 'OC_type_delivery', 'OC_maternal_outcome',
               'Unnamed: 0',
       'OC_pregnancy_outcome', 'OC_birth_delivery_conducted_by',
       'OC_pre_term', 'OC_birth_weight'])

#dropcols.extend(['OC_block_vip','OC_hamlet_name_vip'])
# remove predictors
dropcols.extend([
        'EN_BIRTH_WEIGHT_low_weight_AND_BLANKS',
        'EN_PLACE_BIRTH_at_home_AND_BLANKS',
         'EN_PLACE_BIRTH_in_hospital',   
         'EN_PLACE_BIRTH_at_home',   
         'EN_BIRTH_WEIGHT_normal_weight',
         'EN_PRE_TERM_no', 
         'EN_PRE_TERM_yes',
       'EN_PLACE_BIRTH_in_transit', 'EN_PLACE_BIRTH_nan',
       'EN_TYPE_DELIVERY_---', 'EN_TYPE_DELIVERY_caeserian',
       'EN_TYPE_DELIVERY_nan', 'EN_TYPE_DELIVERY_normal',
       'EN_MATERNAL_OUTCOME_---', 
      'EN_MATERNAL_OUTCOME_maternal_alive',
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
       'EF_tot_TP','OC_SUM_Number pregnancy visits', 
#       'OC_MAX_Months Til EDD',
       'EF_Weeks_Preg_at_Presentation',
       'OC_MIN_lmp_vip', 'OC_block_vip', 'OC_hamlet_name_vip',
       'closedInPeriod-sum'
#       'openedInPeriod-sum',  'anc_2-sum',  'anc_3-sum',  'anc_4-sum',  'tetanus_1-sum',  'takes_nutrition-sum', 'Number pregnancy visits-sum'
#        'EF_openedInPeriod-sum', 'EF_closedInPeriod-sum',  'EF_anc_2-sum',  'EF_anc_3-sum',  'EF_anc_4-sum',  'EF_tetanus_1-sum',  'EF_takes_nutrition-sum'
         ])
df.columns.values
len(dropcols)
df = df.drop(dropcols,axis=1)
df.shape
# =============================================================================
# # PREP TEST FILE
# =============================================================================
train = df
#train = train.drop(['openedInPeriod-sum','closedInPeriod-sum','Number pregnancy visits-sum','EF_anc_2-sum','EF_anc_3-sum','EF_anc_4-sum','EF_tetanus_1-sum','takes_nutrition-sum','OC_block_vip_h','OC_hamlet_name_vip_h'], axis=1)
if (TEST_FILE != ''):
    test_set = test_set_original[train.columns.tolist()]     

for target in TARGETS:
    print("\n\n",target, "=============================================================================")
    train[target] = unified_view[target]
    if (TEST_FILE != ''):
        test_set[target] = test_set_original[target] 
    # =============================================================================
    # MULTI ALGO  CROSS VALIDATION
    # =============================================================================
    # prepare models
    models = []
    models.append(('LogR\t', LogisticRegression()))
#    models.append(('KNN\t', KNeighborsClassifier()))
#    models.append(('DTree\t', DecisionTreeClassifier()))
    models.append(('RandF\t', RandomForestClassifier()))
    models.append(('GBoost\t', GradientBoostingClassifier()))      #COMMENT: operationally expensive, noteably slower than the others
#    models.append(('Dummy\t', DummyClassifier()))     
#    models.append(('D_Stratified', DummyClassifier(strategy='stratified')))  
    models.append(('D_Random', DummyClassifier(strategy='uniform')))   
    models.append(('D_Majority', DummyClassifier(strategy='most_frequent')))
    # CROSS VALIDATION - evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'    
    X = train
    X = X.drop([target], axis=1)  
    Y = train[target]
    X.shape; Y.shape
    seed = 7
    best_model_score = 0
    best_model = 'none'
    print("Cross Validation on Train only")
    print("Model: \t\tcv_mean \tcv_std  \t accuracy \troc_auc \tprecision \trecall  \tf1  \t\tpos_guess\tsupport out of", len(X))
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        
        predicted = model_selection.cross_val_predict(model, X, Y, cv=10)
        accuracy = metrics.accuracy_score(Y, predicted)
        roc_auc = metrics.roc_auc_score(Y, predicted) 
        confus_matrix = metrics.confusion_matrix(Y, predicted)
        pos_guesses= confus_matrix[0,1]+confus_matrix[1,1]
        classification_report = metrics.classification_report(Y, predicted)  
        precision = metrics.precision_score(Y, predicted, pos_label=1)  
        recall = metrics.recall_score(Y, predicted, pos_label=1)  
        f1 = metrics.f1_score(Y, predicted, pos_label=1)
        support = metrics.precision_recall_fscore_support(Y, predicted)
        results.append(cv_results) 
        names.append(name)
        msg = "%s: \t%f \t(%f) \t%f \t%f \t%f \t%f \t%f \t%i \t\t%i" % (name, cv_results.mean(), cv_results.std(), accuracy, roc_auc, precision, recall, f1, pos_guesses,support[3][1])
        print(msg)

        if (roc_auc>best_model_score and 'Dummy' not in str(type(model))):
            best_model_score = roc_auc
            best_model = model
    
    print("best roc = ", best_model_score, type(best_model) )   
    # boxplot algorithm comparison
    if SHOW_GRAPHS:
        fig = plt.figure(figsize = (8,5))
        fig.suptitle('Algorithm Cross-Validation Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show() 

    # =============================================================================
    # FEATURE IMPORTANCE 1
    # =============================================================================
    rf = GradientBoostingClassifier() #assumed best_model 
    rf.fit(X, Y) 
     
    print ("Features sorted by their score: ", target)
    mylist =pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), X), reverse=True))
    mylist.columns = ['feature_importance','feature']
    mylist['cum_importance']= mylist['feature_importance'].cumsum()
    print(mylist[0:15])

    predicted = rf.predict(X) 
    accuracy = metrics.accuracy_score(Y, predicted)
    roc_auc = metrics.roc_auc_score(Y, predicted) 

     # Quick probs curve
    probs = pd.DataFrame(rf.predict_proba(X))      
    probs["id"] = X.index.values
    probs.set_index('id',inplace=True)
    probs[target] = Y #[target]
    probs['IN_age']= X['IN_age']
    probs = probs.sort_values(1, ascending=False)
    probs['count'] = range(1,len(probs)+1)
    probs["risk_cumsum"] = probs[1].cumsum()
    probs["cum_val_pos"] = probs[target].cumsum()
    probs["pct_by_count"] = probs["count"]/len(probs) #7731*0.2 = 1546
    probs["pct_by_risk_cumsum"] = probs["risk_cumsum"]/probs[1].sum()
    probs["pct_by_valpos_cumsum"] = probs["cum_val_pos"]/probs[target].sum()
    if SHOW_GRAPHS:
        probs.plot(x="pct_by_count", y="pct_by_valpos_cumsum", marker='.')
        probs.columns.values
        plt.show()
        percentiles = [1,5,10,20,30,50,70,90,100]
        print("ORDERED BY: probs, TARGET = ",target )
        ranktab = pd.DataFrame()
        for p in percentiles:
            y = probs.loc[probs["count"]==round(len(probs)*p/100),:] 
            y = y.drop([target,'pct_by_risk_cumsum'],axis=1)
            gain = y.iloc[0,len(y.columns)-1]-y.iloc[0,len(y.columns)-2]
            lift = y.iloc[0,len(y.columns)-1]/y.iloc[0,len(y.columns)-2]
#            print(y,"gain=","{:.00%}".format(gain), "or lift", "%.1fx" % lift)
            y['gain'] = "{:.00%}".format(gain)
            y['lift'] = "%.1fx" % lift
            ranktab = ranktab.append(y)
            
        print(ranktab)
        confus_matrix = metrics.confusion_matrix(Y, predicted)
        sns.heatmap(confus_matrix,annot=True,fmt="d")

    # =============================================================================
    # ONE TIME TEST ON TRAINED ALGO
    # =============================================================================
    results = []
    names = []
    x_train = train
    x_train = x_train.drop([target], axis=1)  
    y_train = train[target] #y_train.shape
  
    if (TEST_FILE != ''):
        y_test = test_set[target]  #y_test.shape  
        x_test = test_set.drop([target], axis=1)  
        
        print("TRAIN = ",INPUT_FILE)
        print("TEST = ",TEST_FILE)
        print("Model: \t\taccuracy \troc_auc \tprecision \trecall  \tf1  \t\tpos_guess\tsupport out of", len(y_test))
        for name, model in models:
            model.fit(x_train, y_train)
            predicted = model.predict(x_test) 
    
            accuracy = metrics.accuracy_score(y_test, predicted)
            roc_auc = metrics.roc_auc_score(y_test, predicted) 
            confus_matrix = metrics.confusion_matrix(y_test, predicted)
            pos_guesses= confus_matrix[0,1]+confus_matrix[1,1]
            classification_report = metrics.classification_report(y_test, predicted)  
            precision = metrics.precision_score(y_test, predicted, pos_label=1)  
            recall = metrics.recall_score(y_test, predicted, pos_label=1)  
            f1 = metrics.f1_score(y_test, predicted, pos_label=1)
            support = metrics.precision_recall_fscore_support(y_test, predicted)
            results.append(precision)
            names.append(name)
            msg = "%s: \t%f \t%f \t%f \t%f \t%f \t%i \t\t%i" % (name, accuracy, roc_auc, precision, recall, f1, pos_guesses, support[3][1])
            print(msg)
        

        # =============================================================================
        # FEATURE IMPORTANCE 2
        # =============================================================================
        rf = RandomForestClassifier()  
        rf.fit(x_train, y_train) 
         
        print ("Features sorted by their score: ", target)
        mylist =pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True))
        mylist.columns = ['feature_importance','feature']
        mylist['cum_importance']= mylist['feature_importance'].cumsum()
        print(mylist[0:15])
        
        # Quick probs curve
        probs = pd.DataFrame(rf.predict_proba(x_test))      
        probs["id"] = x_test.index.values
        probs.set_index('id',inplace=True)
        probs = probs.sort_values(1, ascending=False)
        probs['count'] = range(1,len(probs)+1)
        probs["cumsum"] = probs[1].cumsum()
        probs["pct by count"] = probs["count"]/len(probs) #7731*0.2 = 1546
        probs["pct_by_cumsum"] = probs["cumsum"]/probs[1].sum()
        # positives actually in that percentile?
        probs[target] = y_test #[target]
        probs["cum_val_pos"] = probs[target].cumsum()
        probs["pct_by_valpos_cumsum"] = probs["cum_val_pos"]/probs[target].sum()
        if SHOW_GRAPHS:
            probs.plot(x="pct by count", y="pct_by_cumsum", marker='.')
            print(target, "10 percentile")
            print(probs.loc[probs["count"]==round(len(probs)*0.1),:])
            print(target, "20 percentile")
            print(probs.loc[probs["count"]==round(len(probs)*0.2),:])
            # feature importance
            mylist[0:25].plot()
            y = unified_view
            cols = mylist[0:5]['feature'].tolist()
            cols.append(target)
            plt.show()
    #        pplot = sns.pairplot(y[cols], hue=target)  
            confus_matrix = metrics.confusion_matrix(y_test, rf.predict(x_test))
            sns.heatmap(confus_matrix,annot=True,fmt="d")

        # The "accuracy" scoring is proportional to the number of correct classifications
        rfecv = RFECV(estimator=rf, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
        rfecv = rfecv.fit(x_train, y_train)
        print('Optimal number of features :', rfecv.n_features_)
        print('Best features :', x_train.columns[rfecv.support_])
        
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score of number of selected features")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
    
    train.drop([target], axis=1, inplace=True) 
    if (TEST_FILE != ''):
        test_set.drop([target], axis=1, inplace=True)  
    
    
# =============================================================================
# HONESTY BASELINE
# =============================================================================
originals = ['OC_place_birth', 'OC_birth_weight', 'OC_type_delivery']
print("# =============================================================================\n","PREVELANCE IN THE TEST SET")
i=-1
for target in TARGETS:
    print(target)
    if (TEST_FILE != ''):
        print(test_set_original[target].value_counts())     #4375/(4375+1402)
        i+=1
        print(test_set_original[originals[i]].value_counts())  #5541/(5541+268+1503)

unified_view['EF_tot_TP'].describe()
unified_view[unified_view['EF_tot_TP']<6]['OC_place_birth'].value_counts()  #.sum(>10) = 765/9170 =83%

# <6== 2946 ==========================================================================
# EN_PLACE_BIRTH_in_hospital    2339/2946 = 79%
# EN_PLACE_BIRTH_at_home         526/2946 = 17%
# EN_PLACE_BIRTH_---              59/2946 = 2%
# EN_PLACE_BIRTH_in_transit       13/2946
# EN_PLACE_BIRTH_nan               9
# =============================================================================
# >10=============================================================================
# EN_PLACE_BIRTH_in_hospital    462/765 60%
# EN_PLACE_BIRTH_---            189/765 25%
# EN_PLACE_BIRTH_at_home        108/765 14%
# EN_PLACE_BIRTH_in_transit       4/765 1%
# EN_PLACE_BIRTH_nan              2/765 0%
# =============================================================================
# =============================================================================
# END OF FILE
# =============================================================================
print('Execution = ', datetime.datetime.now() - START)

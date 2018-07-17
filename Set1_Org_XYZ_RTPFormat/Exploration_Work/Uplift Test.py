#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:26:13 2018

@author: ksharpey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:37:41 20185

@author: ksharpey
"""
import configparser
import os as os
import numpy as np
import pandas as pd
import math

config = configparser.ConfigParser()
config.read('config.ini')

INPUT_FILE = config['DEFAULT']['INPUT_FILE']
#INPUT_FILE = OUTPUT_DIR +'TrainSet 2 20180310_output_20180313-14h56_EFC_hand encode.csv'
TEST_FILE = config['DEFAULT']['TEST_FILE']
TARGETS = config.get('DEFAULT','TARGETS').split(',')
VERBOSE = config.getboolean('DEFAULT','VERBOSE')

# INITIATE
os.getcwd()
unified_view = pd.read_csv(INPUT_FILE)
test_set_original = pd.read_csv(TEST_FILE) 

# =============================================================================
# UPLIFT POC
# =============================================================================
df = unified_view
#TREATMENTS = ['tetanus_1-RTP1','anc_2-RTP1','Number pregnancy visits-RTP1','takes_nutrition-RTP1']
TREATMENTS = config.get('DEFAULT','TREATMENTS').split(',')
treatment = TREATMENTS[0]
CONF_COEFF = 0.95

uplifts = pd.DataFrame(0, index=TARGETS,columns=TREATMENTS)
uplifts.loc["Treat:ControlRatio"] = np.zeros((1, len(TREATMENTS)))[0].tolist()
uplifts.loc["TreatN"] = np.zeros((1, len(TREATMENTS)))[0].tolist()
uplifts.loc["ControlN"] = np.zeros((1, len(TREATMENTS)))[0].tolist()
uplifts.loc["StdDev_Treat"] = np.zeros((1, len(TREATMENTS)))[0].tolist()
uplifts.loc["StdDev_Sample"] = np.zeros((1, len(TREATMENTS)))[0].tolist()
uplifts.loc["CI"] = np.zeros((1, len(TREATMENTS)))[0].tolist()

for treatment in TREATMENTS:
    if df[treatment].sum()>0 :
        treat = df[df[treatment]!=0]  
        control = df[df[treatment]==0]    
        TCratio = treat[target].count()/control[target].count()
        
        if VERBOSE:     print("TARGET\t \t \t \t",treatment,"\t \t CONTROL\t \t Uplift\t \t \tTreat:ControlRatio")
        for target in TARGETS:
#            RT = treat[target].value_counts()[1]/treat[target].count()
            RT = treat[target].sum()/treat[target].count()
            RC = control[target].sum()/control[target].count()
            if VERBOSE:     print(target,"\t",RT,"\t \t",RC,"\t",RT-RC,"\t",TCratio)
            uplifts.loc[target][treatment] = RT-RC
    
        if VERBOSE:     print("\n")
        uplifts.loc["Treat:ControlRatio"][treatment] = TCratio
        uplifts.loc["TreatN"][treatment] = len(treat)
        uplifts.loc["ControlN"][treatment] = len(control)
        uplifts.loc["StdDev_Treat"][treatment] = treat[target].std()
        uplifts.loc["StdDev_Sample"][treatment] = df[target].std()
        uplifts.loc["CI"][treatment] = CONF_COEFF/2*uplifts.loc["StdDev_Treat"][treatment]/(math.sqrt(treat[target].count()))

print(uplifts)
# =============================================================================
# END UPLIFT POC   
# =============================================================================

output_file_name = INPUT_FILE.rstrip('.csv') + '_UPLIFT3'+'.csv'
uplifts.to_csv(output_file_name)
print(len(uplifts.columns), " uplift written to: ", output_file_name)



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
dropcols.extend(['Unnamed: 0','_id','OC_place_birth', 'OC_type_delivery', 'OC_maternal_outcome',
       'OC_pregnancy_outcome', 'OC_birth_delivery_conducted_by',
       'OC_pre_term', 'OC_birth_weight'])
#TODO these need to be encoded
#dropcols.extend(['OC_block_vip','OC_hamlet_name_vip'])
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
#       'openedInPeriod-sum', 'closedInPeriod-sum',  'anc_2-sum',  'anc_3-sum',  'anc_4-sum',  'tetanus_1-sum',  'takes_nutrition-sum', 'Number pregnancy visits-sum'
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
test_set = test_set_original[train.columns.tolist()]     

train.drop('EF_Weeks_Preg_at_Presentation', axis=1, inplace=True)
test_set.drop('EF_Weeks_Preg_at_Presentation', axis=1, inplace=True)

for target in TARGETS:
    print(target, "=============================================================================")
    train[target] = unified_view[target]
    test_set[target] = test_set_original[target] 
    # =============================================================================
    # MULTI ALGO  CROSS VALIDATION
    # =============================================================================
    # prepare models
    models = []
    models.append(('LogR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DTree', DecisionTreeClassifier()))
    models.append(('RandF', RandomForestClassifier()))
    models.append(('Dummy', DummyClassifier()))
    # CROSS VALIDATION - evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    X = train
    X = X.drop([target], axis=1)  
    Y = train[target]
    X.shape; Y.shape
    seed = 7
    print("Cross Validation on Train only")
    print("Model: \tcv_mean \tcv_std  \t accuracy \troc_auc \tprecision \trecall  \tf1  \t\tsupport out of", len(X))
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        
        predicted = model_selection.cross_val_predict(model, X, Y, cv=10)
        accuracy = metrics.accuracy_score(Y, predicted)
        roc_auc = metrics.roc_auc_score(Y, predicted) 
        confus_matrix = metrics.confusion_matrix(Y, predicted)
        classification_report = metrics.classification_report(Y, predicted)  
        precision = metrics.precision_score(Y, predicted, pos_label=1)  
        recall = metrics.recall_score(Y, predicted, pos_label=1)  
        f1 = metrics.f1_score(Y, predicted, pos_label=1)
        support = metrics.precision_recall_fscore_support(Y, predicted)
    #    print(support[3][1])
        results.append(cv_results)
        names.append(name)
        msg = "%s: \t%f \t(%f) \t%f \t%f \t%f \t%f \t%f \t%i" % (name, cv_results.mean(), cv_results.std(), accuracy, roc_auc, precision, recall, f1, support[3][1])
        print(msg)
        
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # =============================================================================
    # ONE TIME TEST ON TRAINED ALGO
    # =============================================================================
    results = []
    names = []
    x_train = train
    x_train = x_train.drop([target], axis=1)  
    y_train = train[target] #y_train.shape
    y_test = test_set[target]  #y_test.shape  
    x_test = test_set.drop([target], axis=1)  
    print("TRAIN = ",INPUT_FILE)
    print("TEST = ",TEST_FILE)
    print("Model: \taccuracy \troc_auc \tprecision \trecall  \tf1  \t\tsupport out of", len(y_test))
    for name, model in models:
        model.fit(x_train, y_train)
        predicted = model.predict(x_test) 

        accuracy = metrics.accuracy_score(y_test, predicted)
        roc_auc = metrics.roc_auc_score(y_test, predicted) 
        confus_matrix = metrics.confusion_matrix(y_test, predicted)
        classification_report = metrics.classification_report(y_test, predicted)  
        precision = metrics.precision_score(y_test, predicted, pos_label=1)  
        recall = metrics.recall_score(y_test, predicted, pos_label=1)  
        f1 = metrics.f1_score(y_test, predicted, pos_label=1)
        support = metrics.precision_recall_fscore_support(y_test, predicted)
        results.append(precision)
        names.append(name)
        msg = "%s: \t%f \t%f \t%f \t%f \t%f \t%i" % (name, accuracy, roc_auc, precision, recall, f1, support[3][1])
        print(msg)
        
        
    # =============================================================================
    # FEATURE IMPORTANCE
    # =============================================================================
    rf = RandomForestClassifier()  
    rf.fit(x_train, y_train) 
     
    print ("Features sorted by their score: ", target)
    mylist =pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True))
    mylist.columns = ['feature_importance','feature']
    mylist['cum_importance']= mylist['feature_importance'].cumsum()
    mylist[0:25].plot()
    print(mylist[0:15])
    
    y = unified_view
    #y = y[(y['IN_age'] > 0) & (y['IN_age'] <40)] #y.shape
    #y = y[(y['EF_Weeks_Preg_at_Presentation'] > 0) & (y['EF_Weeks_Preg_at_Presentation'] <40)] 
    cols = mylist[0:5]['feature'].tolist()
    cols.append(target)
    pplot = sns.pairplot(y[cols], hue=target)  
    
    train.drop([target], axis=1, inplace=True) 
    test_set.drop([target], axis=1, inplace=True)  
    
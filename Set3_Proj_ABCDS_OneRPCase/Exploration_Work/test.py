#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:35:14 2018

@author: ksharpey
"""

import os as os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn import model_selection
from sklearn import metrics 
from sklearn.metrics import accuracy_score
from  sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


os.getcwd()

df = pd.read_csv('delivery table_encoded_nominus.csv')
df.shape
df.columns.values
df.head()  
# CONFIG
target = 'form.where_born'

#   ENCODE DUMMIES
temp = 'TAR_'+target.upper()[5:]+'_' + df[target].astype(str) 
    # generate dummies column encoding and join
y = pd.get_dummies(temp)  
df = df.join(y) 

# another categorical correlation test
df[target] = df[target].astype('category')
df[target].cat.categories = [2,1,0]
df[target] = df[target].astype('float')

# take a 5% sample as this is computationally expensive
#df_sample = df.sample(frac=0.30)  
# Pairwise plots
#pplot = sns.pairplot(df_sample, hue=target)  
#pplot = sns.pairplot(df_sample, hue="form.where_born", vars=['numVisits.20DP2Birth', 'totalTime.20DP2Birth',
#       'Yes20DP2Birth.f.bp2.play_birth_prep_vid',
#       'Yes20DP2Birth.f.play_family_plan_vid',])  


df.drop(['form.add','form.edd', 'del_completed_time', 'form.safe','Yes20DP2Birth.f.bp2.vehicle'],axis=1,inplace=True)
target = 'form.where_born'
df[target].value_counts(normalize = True) 

df.columns.values
# Join home & blanks
#df['TAR_WHERE_BORN_home_N_other](df['TAR_WHERE_BORN_hospital']-1)**2       #.sum() = 4219

#   BIAS DROP
df.drop([
        #'TAR_WHERE_BORN_home',
         'TAR_WHERE_BORN_hospital',
         'TAR_WHERE_BORN_transit'], axis=1, inplace=True)
df.drop('form.where_born', axis=1, inplace=True)  

target_val = 'TAR_WHERE_BORN_home'
df[target_val].value_counts(normalize = True)
#df.drop([target], axis=1, inplace=True)  

#check cols
for col in df.columns.values:
    print (col, " : ", df[col].iloc[0], type(df[col].iloc[0]))



# Modelling
models = []
models.append(('LogR\t', LogisticRegression()))
models.append(('KNN\t', KNeighborsClassifier()))
models.append(('DTree\t', DecisionTreeClassifier()))
models.append(('RandF\t', RandomForestClassifier()))
#models.append(('GBoost\t', GradientBoostingClassifier()))      #COMMENT: operationally expensive, noteably slower than the others
models.append(('D_Random', DummyClassifier(strategy='uniform')))   
models.append(('D_Majority', DummyClassifier(strategy='most_frequent')))
# CROSS VALIDATION - evaluate each model in turn
results = []
names = []
scoring = 'accuracy'    
X = df
X = X.drop([target_val], axis=1)  
Y = df[target_val]
Y.value_counts(normalize=True)
seed = 7
print("Cross Validation on Train only, target_val = ", target_val)
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

Y.value_counts(normalize = True) 

 # =============================================================================
# FEATURE IMPORTANCE for train set
# =============================================================================
rf = RandomForestClassifier() 
#rf = GradientBoostingClassifier()
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
rf.fit(x_train, y_train) 
 
print ("\nFeatures sorted by their score: ", target," = ", target_val)
mylist =pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True))
mylist.columns = ['feature_importance','feature']
mylist['cum_importance']= mylist['feature_importance'].cumsum()
print(mylist[0:15])

rf_model = rf.fit(x_train, y_train)  
# training accuracy 99.74%
rf_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(x_test))  
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# Store metrics
rf_accuracy = metrics.accuracy_score(y_test, predicted)  
rf_roc_auc = metrics.roc_auc_score(y_test, probs[1])  
rf_confus_matrix = metrics.confusion_matrix(y_test, predicted)  
rf_classification_report = metrics.classification_report(y_test, predicted)  
rf_precision = metrics.precision_score(y_test, predicted, pos_label=1)  
rf_recall = metrics.recall_score(y_test, predicted, pos_label=1)  
rf_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print("rf_accuracy: ",rf_accuracy, ", rf_roc_auc: ", rf_roc_auc)
print("confusion matrix \n", rf_confus_matrix)
print("prevelence in y_test:")
print(y_test.value_counts(normalize = True))

# Evaluate the model using 10-fold cross-validation
rf_cv_scores = cross_val_score(
        RandomForestClassifier()
        , x_test, y_test, scoring='precision', cv=10) 
rf_cv_mean = np.mean(rf_cv_scores)  
 

# =============================================================================
# TEST A BALANCED SVM
# =============================================================================
# create a smaller sample
X[target_val] = Y
X.shape
sample = X.sample(frac=0.1, replace=False)
Y = sample[target_val] 
sample = sample.drop([target_val], axis=1)  
sample.shape
print("(pre-balance) incidence of target = ", target_val)
print(Y.value_counts(normalize=True))

# Train model
clf_3 = SVC(kernel='linear', 
            class_weight='balanced', # penalize
            probability=True)

clf_3.fit(sample, Y)
 
# Predict on training set
pred_y_3 = clf_3.predict(sample)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_3 ) )
# [0 1]
print("(post-balance) target predictions = ", target_val)
print(pd.Series(pred_y_3).value_counts(normalize=True))
 
# How's our accuracy?
print("SVM Accuracy:", accuracy_score(Y, pred_y_3) )
# 0.609
 
# What about AUROC?
prob_y_3 = clf_3.predict_proba(sample)
prob_y_3 = [p[1] for p in prob_y_3]
print("SVM ROCAUC:", roc_auc_score(Y, prob_y_3) )
# 0.5305236678



#!/usr/bin/python

import sys
import pickle
import pandas as pd
import pprint as pp
import numpy as np
import tester
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

print '\n\n===========================================Task 1: Select features ==========================================\n'


stock_features = ['exercised_stock_options',
                  'restricted_stock',
                  'restricted_stock_deferred',
                 'total_stock_value'
                 ]

non_stock_features = ['salary',
                    'bonus',
                   'long_term_incentive',
                   'deferred_income',
                   'deferral_payments',
                    'loan_advances',
                   'other',
                   'expenses',                
                    'director_fees', 
                    'total_payments']

email_features = ['from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi',
                  'to_messages']

new_features = ['to_poi_ratio',
                'from_poi_ratio',
                'shared_poi_ratio',
                'bonus_to_salary',
                'bonus_to_total'
               ]


features_list =  ['poi'] + stock_features  + non_stock_features + email_features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
data_df = pd.DataFrame(data_dict).transpose()
data_df = data_df[features_list]

print "\n"
print "Names of Features: ",features_list
print "\n"
print "Number of cloumns: ", data_df.shape[1]
print "\n"
print "Number of rows: " , data_df.shape[0]

print "\n"

   
### All features except email address could belong to int data type, also email address is reduntant fetaure as all names or keys are unique

## There were two names for keys which were not looked as real names one is "THE TRAVEL AGENCY IN THE PARK" and the otehr is "TOTAL"

from pprint import pprint

print "Outliers Salary \n"
outliers_salary = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers_salary.append((key,int(val)))
    
pprint(sorted(outliers_salary,key=lambda x:x[1],reverse=True)[:10])

print "\n"
print "Outliers Bonus \n"
outliers_bonus = []
for key in data_dict:
    val1 = data_dict[key]['bonus']
    if val1 == 'NaN':
        continue
    outliers_bonus.append((key,int(val1)))
    
print "\n\n"
pprint(sorted(outliers_bonus,key=lambda x:x[1],reverse=True)[:10])


print "\n\n"
print "===========================================Number of POI========================================================="

counter = 0
for key in data_dict:
    poi_number = data_dict[key]['poi']
    if poi_number == True:
        counter += 1
             
print counter

print "\n"

### Task 2: Remove outliers

print "=======================================Task 2 : Remove Outliers and Data Cleaning=======================================\n"
print data_df.axes[0].tolist()

print "\n\n"

print data_df.axes[1].tolist()

print "\n\n"

## with further analysis "TOTAL" has the highest salary and bonus but poi as "False" whereas for "THE TRAVEL AGENCY IN THE PARK" 
## "poi" false also all other fields as NaNs.

data_df.drop(axis=0, labels = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)

print "\n"
print "Row  THE TRAVEL AGENCY IN THE PARK and TOTAL are removed from the dataset"
print "\n"

## Remove rows which have only NaN values
counter = 0
for k,v in data_dict.iteritems():
    for keys, values in v.iteritems():       
        if values  == 'NaN':
            counter = counter + 1
            if counter == 20:
                print k

## k = ELLIOTT STEVEN who will be removed from the data set  
data_df.drop(axis = 0, labels = ['ELLIOTT STEVEN'],inplace = True)              
print "ELLIOTT STEVEN is removed from the dataset"

### with the anlysis above about Restricted_Stock_Deferred and Restricted Stock for Poi and Non poi's, it is visible
## that this variable is not very effective to know about Poi's and non poi's as most poi's and non poi's both have NaN
## NaN values except one for HIRKO JOSEPH for whom restricted stocks are NaN.


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

print "\n\n=================================Task 3: Create New Features =========================================\n\n"
#my_dataset = data_dict
data_df = data_df.replace('NaN', np.nan)
data_df[features_list] = data_df[features_list].fillna(value=0)


# Add in additional features to dataframe
data_df['to_poi_ratio'] = data_df['from_poi_to_this_person'] /data_df['to_messages']
data_df['from_poi_ratio'] = data_df['from_this_person_to_poi'] / data_df['from_messages']
data_df['shared_poi_ratio'] = data_df['shared_receipt_with_poi'] / data_df['to_messages']
data_df['bonus_to_salary'] = data_df['bonus'] / data_df['salary']
data_df['bonus_to_total'] = data_df['bonus'] /data_df['total_payments']



features_list =  ['poi']+stock_features  + non_stock_features + email_features + new_features
data_df.fillna(value = 0, inplace=True)



print data_df.axes[1].tolist()
features_list = data_df.axes[1].tolist()


print "\n\n===============New Features added==================================================================\n\n"
my_dataset = data_df.to_dict(orient='index')

from sklearn.feature_selection import SelectKBest
def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

features_selected = get_k_best(my_dataset,features_list,k=18)
print features_selected


### Extract features and labels from dataset for local testing

from sklearn.feature_selection import SelectKBest
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.metrics import accuracy_score


print "--------------------------------------------Before Tuning----------------------------------------------------------\n\n"

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

    
print "--------------------------------------------Naive Bayes Classifier----------------------------------------------\n\n"
##### Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

print "--------------------------------------------Decision Tree Classifier------------------------------------------------\n\n"
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)  
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()



print "--------------------------------------------Random Forest Classifier------------------------------------------------\n\n"
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, criterion="entropy")
clf.fit(features_train, labels_train)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()


    
print "--------------------------------------------Support Vector Machine Classifier------------------------------------------------\n\n"
from sklearn.svm import SVC
clf = SVC()
clf.fit(features_train, labels_train)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()



print "--------------------------------------------AdaBoost Classifier------------------------------------------------\n\n"
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
print "\n--------------------------------After Tuning-------------------------------------------------\n"

print '\n----------------------------Pipeline, GridSearchCV, and StratifiedShuffleSplit-------------------------------------\n'
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

### StratifiedShuffleSplits for 1000 internal cross-validation splits
### within the grid-search.
sss = StratifiedShuffleSplit(labels,test_size = 0.1,train_size = .1, n_iter=100)
param_grid = {'model__splitter':['best', 'random'],
              'model__min_samples_split':[2,4,6,10],
              'model__criterion':['gini', 'entropy'],
              'model__random_state':[42],
              'model__max_depth':[None,1,2,3,4]
              }
# Decision Tree classifier
dt_tree = tree.DecisionTreeClassifier()

# Pipeline object. I have used all the steps, it's up to you if you want to use all or just use algorithm tuning.
pipe = Pipeline(steps=[('minmaxer', preprocessing.MinMaxScaler()),
                       ('select', SelectKBest(k=18)), 
                       ('pca', PCA(n_components = 2)), 
                       ('model', dt_tree)
                       ])
gs = GridSearchCV(pipe, param_grid=param_grid, cv=sss,
                             scoring = 'f1')
gs.fit(features, labels)
clf = gs.best_estimator_


### Score of best_estimator on the left out data
print "best score is {0}".format(gs.best_score_)

# Print the optimized parameters used in the model selected from grid search
print "Params: ", gs.best_params_ 


tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()




print "\n\n---------------------------------AdaBoost Tuning Ensembling---------------------------------------------------------\n\n"


pca = PCA()
cv = StratifiedShuffleSplit(labels, 100, test_size=0.1, train_size=0.1, random_state=42)
DTC =  DecisionTreeClassifier(criterion="gini", splitter="best", 
                              max_features="auto", random_state=42,
                              min_samples_split = 2,
                              max_depth = None)
clf = AdaBoostClassifier()

pipe = Pipeline(steps=[('scaling',preprocessing.MinMaxScaler()),
                       ('select', SelectKBest(k=18)), 
                       ('pca',PCA(n_components = 2)),
                       ('ada_boost', clf)
                      ])

parameters = {"pca__random_state":[60],
              "ada_boost__base_estimator": [DTC],
              "ada_boost__n_estimators": [10,50,100,150,200,500,700,1000,1200],
              "ada_boost__learning_rate": [[0.1,0.5,0.75,1.0,2.0,3.0]],
              "ada_boost__algorithm" : ["SAMME", "SAMME.R"],
              "ada_boost__random_state" : [42]
             }

grid = GridSearchCV(pipe, parameters, cv=cv,scoring = 'f1')
grid.fit(features, labels)
pp.pprint(grid.best_params_)

clf = grid.best_estimator_
#pred = clf.predict(features_test)
#print clf
#print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

print "\n\n Adaboost Params: ", grid.best_params_ ,"\n\n"
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

### Assign the best estimator to final SVM classifier

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

sys.stdout.close()
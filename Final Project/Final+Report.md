
### Question Responses for the "Identifying Enron Fraud" Project
#### By Niharika Jeena Shah
#### Date: 06/02/2018


#### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

##### Data Exploration

The goal of this project is to use machine learning techniques in order to know if we are identifying correct person of interests(poi) in the given dataset. The given dataset has 146 keys or people who worked for Enron and 21 features associated with these key people. Out of 146 key people there were 18 POI's in the dataset. We used python scikit-learn package to create machine learning models in to know if given the new data, which would be passed thorugh the model, would predict the POI's with accuracy, precision, and recall. The features given in the dataset can be divided in three main categories:

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (except email_address all other features in this category are numbers)

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

poi label = [‘poi’] (boolean, represented as integer)

The person of interest are those people who were found tried for fraud and criminal activity in Enron fraud case. The objective of this project is seperate out POI's and Non POI's using machine learning algorithms.

The dataset is available at [Udacity Enron Dataset](https://github.com/udacity/ud120-projects)


##### Outlier Investigation and Data Cleaning
I removed key "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" as both of these do not refer to a person. I had second thoughts about "THE TRAVEL AGENCY IN THE PARK" for not removing it from the dataset, however; this was not one of 'POI' therefore I believed that this key can be removed safely from the given dataset. Also, "THE TRAVEL AGENCY IN THE PARK" has all fiels as NaNs.
I also dropped key values which had all NaN values in the dataset, one of which was 'ELLIOTT STEVEN'



####  What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

##### Create New Features

I divided all features in three categories excluding 'poi'. 
email_features = ['from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi',
                  'to_messages']
stock_features = ['exercised_stock_options',
                  'restricted_stock',
                  'restricted_stock_deferred',
                  'total_stock_value']
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
I created 'to_poi_ratio','shared_poi_ratio', and 'from_poi_ratio' using email_features and 'bonus_to_salary','bonus_to_total' from non_stock_features, and 'exercised_to_total' from stock_features
new_features = ['to_poi_ratio',
                'from_poi_ratio',
                'shared_poi_ratio',
                'bonus_to_salary',
                'bonus_to_total']
                             
                
                
The next step is to create new features from the existing information that could possibly improve performance. I will also need to carry out feature selection to remove those features that are not useful for predicting a person of interest.
After thinking about the background of the Enron case and the information to work with contained in the dataset, I decided on three new features to create from the email metadata. The first will be the ratio of emails to an individual from a person of interest to all emails addressed to that person, the second is the same but for messages to persons of interest, and the third will be the ratio of email receipts shared with a person of interest to all emails addressed to that individual. The rational behind these choices is that the absolute number of emails from or to a person of interest might not matter so much as the relative number) considering the total emails an individual sends or receives. My instinct says that individuals who interact more with a person of interest (as indicated by emails) are themselves more likely to be a person of interest because the fraud was not perpertrated alone and required a net of persons. However, there are also some innocent persons who may have sent or received many emails from persons of interest simply in the course of their daily and perfectly abovethe-table work.


##### Feature Selection

[Feature selection](https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/) is a process where you automatically select those features in your data that contribute most to the prediction variable or output in which you are interested.
Having too many irrelevant features in your data can decrease the accuracy of the models. Three benefits of performing feature selection before modeling your data are:
* Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
* Improves Accuracy: Less misleading data means modeling accuracy improves.
* Reduces Training Time: Less data means that algorithms train faster.

After many iterations for after tuning algorithms, many were constant and gave userwarning such as
UserWarning: Features [1] are constant. After reading up online about userwarnings for contant features I removed most of features to improve my model's performance. However: selecting features manually didn't improve performance of selected algorithms.
Going back to all features along with new features and selecting 18 best features out of features using SelectKBest gave me precision and recall rates above .3, however; I suspect that this would be a case of overfitting of the given data.

I also implemented SelectKBest algorithm and manually passed for selecting top 18 features from features_list which are:
 ['salary', 'bonus_to_total', 'other', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'exercised_stock_options', 'bonus_to_salary', 'expenses', 'shared_poi_ratio', 'to_poi_ratio', 'deferred_income', 'from_poi_ratio', 'restricted_stock', 'long_term_incentive']
 


##### Feature Scaling
For the final model,I did use Pipelines combined with GridSearchCV to search over parameters for all processing steps at once for scaling and preprocessing of the given dataset.

####  What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
I tried Naive Bayes, Decision Tree, Random Forest, SVM, and Adaboost classifiers for testing the given dataset.
The performances for each model are :
#### Before Tuning
##### Naive Bayes
 Accuracy: 0.73780       Precision: 0.21880      Recall: 0.37600 F1: 0.27662     F2: 0.32876
 
 
##### Decision Tree
Accuracy: 0.80120       Precision: 0.25450      Recall: 0.25450 F1: 0.25450     F2: 0.25450


##### Random Forest
 Accuracy: 0.86040       Precision: 0.42370      Recall: 0.13050 F1: 0.19954     F2: 0.15146
 
 
##### SVM
Precision or recall may be undefined due to a lack of true positive predicitons.


##### AdaBoost
 Accuracy: 0.84547       Precision: 0.40136      Recall: 0.32350 F1: 0.35825     F2: 0.33656
 
 
#### After Tuning with Preprocessing and Scaling using Pipelines and GridSearch

##### Decision Tree
Accuracy: 0.84053       Precision: 0.35219      Recall: 0.23350 F1: 0.28082     F2: 0.25038


##### AdaBoost
 Accuracy: 0.83347       Precision: 0.35607      Recall: 0.30800 F1: 0.33029     F2: 0.31655
 
 
 
I chose AdaBoost over other algorithms as for tester.py this algorithm gave the best performance if precision and recall rate are considered for performance of an algorithm.

####  What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Machine learning models are parameterized so that their behavior can be tuned for a given problem. Models can have many parameters and finding the best combination of parameters can be treated as a search problem. Algorithm tuning is a final step in the process of applied machine learning before presenting results. It is sometimes called Hyperparameter optimization where the algorithm parameters are referred to as hyperparameters whereas the coefficients found by the machine learning algorithm itself are referred to as parameters. Optimization suggests the search-nature of the problem. Phrased as a search problem, you can use different search strategies to find a good and robust parameter or set of parameters for an algorithm on a given problem. Two simple and easy search strategies are grid search and random search. Scikit-learn provides these two methods for algorithm parameter tuning and examples of each are provided below.

I used GridSearchCV which is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.

f_claasif in SelectKbest kept giving error RuntimeWarning: invalid value encountered in divide f = msb / msw, therefore changes SelectKBest parameter score_func from f_classif to chi2. chi2 works only for positive data,and in the given dataset after initial analysis all of the data is positive, hence chi2 can be used safely as an option. I ended up manually selecting the number of features passed in SelectKBest, both in DecisionTreeClassifier and AdaBoost Classifier and it worked best with 18 features.

Scoring = 'f1' was used in GridSearchCv as this is value which is used generally for ‘f1’ for binary targets,

With Adaboost algorithm I tried n_estimators with [10,50,100,150,200,500,700,1000,1200] and learning rate  [[0.1,0.5,0.75,1.0,2.0,3.0]] and best n_estimators and learning rate were and the algorithm selected was SAMME.

Whereas when I used default AdaBoostClassifier without passing any manual values for paramaeters I have got algorthm as SAMME.R, n_estimators as 50 and best learning rate as 1.0 which is default learning rate for AdaBoostClassifier. 

####  What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

[Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29), sometimes called rotation estimation, or out-of-sample testing is any of various similar model validation techniques for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. In a prediction problem, a model is usually given a dataset of known data on which training is run (training dataset), and a dataset of unknown data (or first seen data) against which the model is tested (called the validation dataset or testing set).[4] The goal of cross-validation is to test the model’s ability to predict new data that were not used in estimating it, in order to flag problems like overfitting[citation needed] and to give an insight on how the model will generalize to an independent dataset (i.e., an unknown dataset, for instance from a real problem).


I decided to go with StratifiedShuffleSplit because; StratifiedShuffleSplit is a variation of ShuffleSplit, which returns stratified splits, i.e which creates splits by preserving the same percentage for each target class as in the complete set.

While using StratifiedShuffleSplit I tried various test_size and train_size for the dataset in order to get expected precision and recall rates. 
Added train_size and test_size .5,.3.,and .1 in StratifiedShuffleSplit due to this warning. However, it did not improve any results. For DecisionTreeClassifier it improved precision rate but not recall rate. However, for adaboost it did not change at all.
While using GridSearchCV parameter verbose made a lot more sense to see how algorithms are working behind the scenes.



##### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]



###### -------------------------------------------------Before Tuning---------------------------------------------------------------------------------------------------

AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
        Accuracy: 0.84580       Precision: 0.40298      Recall: 0.32500 F1: 0.35981     F2: 0.33808
        Total predictions: 15000        True positives:  650    False positives:  963   False negatives: 1350   True negatives: 12037
        
##### --------------------------------------------------After Tuning-------------------------------------------------------------------------------------------------      

###### ------------DecisionTree Classifier----------

Params:  {'model__splitter': 'random', 'model__criterion': 'gini', 'model__max_depth': None, 'model__min_samples_split': 2, 'model__random_state': 42}
Pipeline(memory=None,
     steps=[('minmaxer', MinMaxScaler(copy=True, feature_range=(0, 1))), ('select', SelectKBest(k=18, score_func=<function f_classif at 0x000000000D6C5E48>)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('model', DecisionT...      min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='random'))])
        Accuracy: 0.82213       Precision: 0.31588      Recall: 0.28650 F1: 0.30047     F2: 0.29193
        Total predictions: 15000        True positives:  573    False positives: 1241   False negatives: 1427   True negatives: 11759

 Accuracy: 0.84053       Precision: 0.35219      Recall: 0.23350 F1: 0.28082     F2: 0.25038
        Total predictions: 15000        True positives:  467    False positives:  859   False negatives: 1533   True negatives: 12141
###### ------------Adaboost------------------

{'ada_boost__algorithm': 'SAMME',
 'ada_boost__base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'),
 'ada_boost__learning_rate': [0.1, 0.5, 0.75, 1.0],
 'ada_boost__n_estimators': 10,
 'ada_boost__random_state': 42,
 'pca__random_state': 60}
Pipeline(memory=None,
     steps=[('scaling', MinMaxScaler(copy=True, feature_range=(0, 1))), ('select', SelectKBest(k=18, score_func=<function f_classif at 0x000000000D6C5E48>)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=60,
  svd_solver='auto', tol=0.0, whiten=False)), ('ada_boost', AdaBoost...best'),
          learning_rate=[0.1, 0.5, 0.75, 1.0], n_estimators=10,
          random_state=42))])
        Accuracy: 0.83347       Precision: 0.35607      Recall: 0.30800 F1: 0.33029     F2: 0.31655
        Total predictions: 15000        True positives:  616    False positives: 1114   False negatives: 1384   True negatives: 11886
        
        Accuracy: 0.83347       Precision: 0.35607      Recall: 0.30800 F1: 0.33029     F2: 0.31655
        Total predictions: 15000        True positives:  616    False positives: 1114   False negatives: 1384   True negatives: 11886
        
        
A precision score of 0.35607 means that of the individuals labeled by my model as persons of interest, 35.6% of them were indeed persons of interest. A recall score of 0.30800 means that my model identified 30.8% of persons of interest present in the entire dataset.

Adaboost Classifier gives better precision and recall rate before tuning which for precision and recall are  0.40298 and 0.32500 respectively.

##### Takeaway from this project

An important learning from this project is manual finetuning of parameters. Automatic selection of parameters and their values does not give always best results. 


#### References

http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection_pipeline.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-pipeline-py

http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py

http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

https://stackoverflow.com/questions/45321435/parameter-tuning-using-gridsearchcv

https://www.researchgate.net/application.TemporarilyBlocked.html

https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/

https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/


```python

```

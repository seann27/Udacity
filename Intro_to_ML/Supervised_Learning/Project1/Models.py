#!/usr/bin/python3
'''
Module that evaluates the performance of a group of supervised learning models
against a dataset split up into training and testing data. Each model uses default
parameters for making the assessment. Supplmental graphing and text results are generated
whenever model performance is evaluated. The model with the best F-Score on the testing
data is returned as the best classifier.
'''

from time import time
# import models
from sklearn.ensemble import \
    RandomForestClassifier, \
    GradientBoostingClassifier, \
    AdaBoostClassifier, \
    BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

# import metrics
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import GridSearchCV

classifiers = {}

def train_predict(learner,X_train,y_train,X_test,y_test,fbeta):
    results = {}
    start = time()
    learner.fit(X_train,y_train)
    end = time()
    results['train_time'] = end-start

    # grab 25% training data for validation
    val = int(0.1*len(X_train))

    start = time()
    predictions_validation = learner.predict(X_train[:val])
    predictions_test = learner.predict(X_test)
    end = time()
    results['prediction_time'] = end-start

    results['acc_score_train'] = accuracy_score(y_train[:val],predictions_validation)
    results['f_score_train'] = fbeta_score(y_train[:val],predictions_validation,beta=fbeta)
    results['acc_score_test'] = accuracy_score(y_test,predictions_test)
    results['f_score_test'] = fbeta_score(y_test,predictions_test,beta=fbeta)

    return results

def evaluate_models(X_train,y_train,X_test,y_test,fbeta=1):
    # initalize models
    classifiers['RandomForest']=RandomForestClassifier(random_state=0,n_estimators=100)
    classifiers['GradientBoost']=GradientBoostingClassifier(random_state=0)
    classifiers['AdaBoost']=AdaBoostClassifier(random_state=0)
    classifiers['Bagging']=BaggingClassifier(random_state=0)
    classifiers['DecisionTree']=DecisionTreeClassifier(random_state=0)
    classifiers['SVC']=SVC(random_state=0,gamma='auto')
    classifiers['GaussianNB']=GaussianNB()
    classifiers['KNeighbors']=KNeighborsClassifier()
    classifiers['SGDC']=SGDClassifier(random_state=0,max_iter=5,tol=0.001)
    classifiers['LogisticRegression']=LogisticRegression(random_state=0)

    # get results from fitting and predicting on each model
    # write files used for generating graphs
    file = open("model_eval_results.txt","w")
    graph = open("model_eval_graph_data.txt","w")
    y_values = []
    x1_values = []
    x2_values = []
    x3_values = []
    x4_values = []
    x5_values = []
    x6_values = []

    fscore_testing = 0
    best_clf = None

    for key,value in classifiers.items():
        print("Evaluating {} . . .".format(key))
        results = train_predict(value,X_train,y_train,X_test,y_test,fbeta)
        if(results['f_score_test'] > fscore_testing):
            fscore_testing = results['f_score_test']
            best_clf = value
        file.write("Model: {}\n".format(key))
        file.write("Accuracy - Train: {}\n".format(results['acc_score_train']))
        file.write("Accuracy - Test: {}\n".format(results['acc_score_test']))
        file.write("F-Score - Train: {}\n".format(results['f_score_train']))
        file.write("F-Score - Test: {}\n".format(results['f_score_test']))
        file.write("Time - Train: {}\n".format(results['train_time']))
        file.write("Time - Predict: {}\n\n".format(results['prediction_time']))
        y_values.append(key)
        x1_values.append(results['acc_score_train'])
        x2_values.append(results['acc_score_test'])
        x3_values.append(results['f_score_train'])
        x4_values.append(results['f_score_test'])
        x5_values.append(results['train_time'])
        x6_values.append(results['prediction_time'])

    def write_graph_settings(list):
        for item in list:
            graph.write(str(item)+",")
        graph.write("\n")

    write_graph_settings(y_values)
    write_graph_settings(x1_values)
    write_graph_settings(x2_values)
    write_graph_settings(x3_values)
    write_graph_settings(x4_values)
    write_graph_settings(x5_values)
    write_graph_settings(x6_values)

    return best_clf

def optimize_best_model(classifier,X_train,y_train,X_test,y_test):
    param_grid = []
    name = classifier.__class__.__name__
    if(name == 'GradientBoostingClassifier'):
        param_grid.append({
            'learning_rate':[0.1,0.5,1,1.5],
            'n_estimators':[100,150,200,300]
        })
        param_grid.append({
            'min_samples_split':[2,4,6,8],
            'min_samples_leaf':[1,2,4,6],
            'max_depth':[3,5,7]
        })
    elif(name == 'RandomForestClassifier'):
        param_grid.append({
            'criterion':['gini','entropy'],
            'min_samples_split':[2,4,6,8],
            'min_samples_leaf':[1,2,4,6],
            'max_depth':[3,5,7]
        })
    elif(name == 'AdaBoostClassifier'):
        param_grid.append({
            'learning_rate':[0.1,0.5,1,1.5],
            'n_estimators':[100,150,200,300],
            'algorithm':['SAMME','SAMME.R']
        })
    elif(name == 'BaggingClassifier'):
        param_grid.append({
            'n_estimators':[100,150,200,300],
            'max_samples':[0.5,1,1.5,2],
            'bootstrap':[True,False]
        })
    elif(name == 'DecisionTreeClassifier'):
        param_grid.append({
            'criterion':['gini','entropy'],
            'min_samples_split':[2,4,6,8],
            'min_samples_leaf':[1,2,4,6],
            'max_depth':[3,5,7]
        })
    elif(name == 'SVC'):
        param_grid.append({

        })
    elif(name == 'GaussianNB'):
        param_grid.append({

        })
    elif(name == 'KNeighborsClassifier'):
        param_grid.append({

        })
    elif(name == 'SGDClassifier'):
        param_grid.append({

        })
    elif(name == 'LogisticRegression'):
        param_grid.append({

        })

    grid_obj = GridSearchCV(classifier,param_grid,n_jobs=3,verbose=5)
    grid_fit = grid_obj.fit(X_train,y_train)
    best_clf = grid_fit.best_estimator_
    return best_clf

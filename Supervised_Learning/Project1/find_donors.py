#!/usr/bin/python3

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score,make_scorer
import Models

# Load the Census dataset
data = pd.read_csv('Udacity_Materials/census.csv')

# separate features from labels
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# transform skewed data
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Normalize numerical features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# One-hot encode categorical features
features_final = pd.get_dummies(features_log_minmax_transform)
income = income_raw.map({'>50K': 1, '<=50K': 0})

# Shuffle and Split data
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Evaluate Model Performance with fbeta = 0.5
fbeta = 0.5
best_clf = Models.evaluate_models(X_train,y_train,X_test,y_test,fbeta)
print("\n",best_clf.__class__.__name__)
best_clf = Models.optimize_best_model(best_clf,X_train,y_train,X_test,y_test)
model_predictions = best_clf.predict(X_test)
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, model_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, model_predictions, beta = 0.5)))

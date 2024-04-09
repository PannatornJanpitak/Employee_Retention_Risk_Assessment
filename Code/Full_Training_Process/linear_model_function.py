
"""

This file contain all function for creating linear model for [Data Science Job Salaries 2020 - 2024] dataset

"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from joblib import dump

#model parameter
def define_model_params():
    """
    #This function contain linear model parameter to do gridsearch in fine-tuning process
    #1.Linear Regression model
    #2.Ridge Regression model
    #3.Lasso Regression model
    #4.ElasticNets Regression model
    #5.Random Forest Regression model
    #6.Support Vector Machine Regression model
    input = []
    output = model parameters in this function
    """
    model_params = {
    'Logistic': {
        'model': LogisticRegression(),
        'params': {
            'penalty': ['l1', 'l2'],
            'C': [50, 100, 150]
            }
        },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(),  
        'params': {
            'loss': ['log_loss', 'exponential'],
            'learning_rate': [0.008, 0.1, 0.3],
            'n_estimators': [150, 200, 250],
            'criterion': ['friedman_mse', 'squared_error']
            }
        },
    'KNeighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [7, 10, 13],
            'weights': ['uniform','distance'],
            'algorithm': ['auto', 'ball_tree','kd_tree', 'brute']

            }
        },
    'RandomForest': {
        'model': RandomForestClassifier(), 
        'params': {
            'n_estimators': [150, 200, 250],
            'criterion': ['gini', 'entropy”', 'log_loss']
            }
        },
    'SVM': {
        'model': SVC(), 
        'params': {
            'C': [50, 100, 150],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            }
        }
    }
    return model_params


#train machine learning model with Gridsearch
def find_best_linear_model(X, y):
    """
    Performs grid search for fine-tuning linear models.
    
    Args:
    - X: Features
    - y: Target
    
    Returns:
    - df_result: Dataframe of model scores and best parameters
    - linear_models: Dictionary of best linear models
    """
    #Get model parameters
    model_params = define_model_params()
    results = []
    linear_models = {}

    #Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    for model_name, model_param in model_params.items():
        #GridSearch
        gs = GridSearchCV(model_param['model'], model_param['params'], cv=5, return_train_score=False)
        gs.fit(X_scaled, y)
        #record result
        linear_models[model_name] = gs.best_estimator_
        results.append({
            "model": model_name,
            "best_score": gs.best_score_,
            "best_params": gs.best_params_
        })
    #convert gridsearch result to dataframe    
    df_result = pd.DataFrame(results, columns=['model', 'best_score', 'best_params'])

    #save best linear models
    save_best_linear_model(df_result, linear_models)

    return  df_result, linear_models

#Save best linear model
def save_best_linear_model(df_result, linear_models):
    best_linear_model =  linear_models[df_result.model[df_result.best_score.idxmax()]]
    dump(best_linear_model, 'linear_best_model.pkl')


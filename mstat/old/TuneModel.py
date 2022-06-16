# coding: utf-8
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import Isomap
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib import projections, pyplot as plt
import sys, os
from datetime import *

from dependencies.ms_data.MSFileReader import MSFileReader
from dependencies.ms_data.AnalysisVis import AnalysisVis
from dependencies.ms_data.MSDataAnalyser import MSDataAnalyser
    
    
help_message = """
Console Command: python tuneModel.py <path/file_name.csv> <random_search>
Arguments:
    <path/file_name.csv>      - (String) path and name of CSV file including the extension ".csv"
    <random_search>           - (Boolean)"""

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def pcaldaTune(rnd_state):
    random_grid = {
        'pca__n_components': [10,20,30,40,50],
        'pca__whiten': [True, False],
        'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
        'lda__solver': ['svd', 'lsqr', 'eigen']
    }

    param_grid = {
        'pca__n_components': [38,39,40,41,42],
        'pca__whiten': [True, False],
        'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
        'lda__solver': ['svd', 'lsqr', 'eigen']
    }

    steps = [
        ('pca', PCA(random_state=rnd_state)),
        ('lda', LDA())
    ]
    est = Pipeline(steps)

    return random_grid, param_grid, est

def pcaqdaTune(rnd_state):
    random_grid = {
        'pca__n_components': [5,10,20,30,40,50],
        'pca__whiten': [True, False],
        'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
        'qda__reg_param': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    }

    param_grid = {
        'pca__n_components': [4,5,6,7,8],
        'pca__whiten': [True, False],
        'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
        'qda__reg_param': [0.0, 0.01, 0.05]
    }

    steps = [
        ('pca', PCA(random_state=rnd_state)),
        ('qda', QDA())
    ]
    est = Pipeline(steps)

    return random_grid, param_grid, est

def rfcTune(rnd_state):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    
    random_grid = {'rfc__n_estimators': n_estimators,
                'rfc__max_features': max_features,
                'rfc__max_depth': max_depth,
                'rfc__min_samples_split': min_samples_split,
                'rfc__min_samples_leaf': min_samples_leaf,
                'rfc__bootstrap': bootstrap}

    param_grid = {'rfc__n_estimators': [int(x) for x in np.linspace(start = 300, stop = 500, num = 10)],
                'rfc__max_features': ['sqrt'],
                'rfc__max_depth': [int(x) for x in np.linspace(10, 30, num = 11)],
                'rfc__min_samples_split': [8, 10, 12],
                'rfc__min_samples_leaf': [2, 3],
                'rfc__bootstrap': [False]}

    steps = [
        ('rfc', RandomForestClassifier(random_state=rnd_state))
    ]
    est = Pipeline(steps)

    return random_grid, param_grid, est

def svcTune(rnd_state):
    # {'svc__kernel': 'linear', 'svc__gamma': 0.001, 'svc__C': 1, 'pca__n_components': 90}
    random_grid = {
        'svc__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': [1,0.1,0.01,0.001],
        'iso__n_components': [10,30,50,70,90,110,130]
        }

    param_grid= {
        'svc__kernel': ('linear', 'rbf'),
        'svc__C': [0.0005,0.001,0.005],
        'svc__gamma': [0.000001,0.00001],
        'iso__n_components': [75,80,85]
        }

    steps = [
        ('sscl', StandardScaler()),
        ('iso', Isomap()),
        ('svc', SVC(random_state=rnd_state))
    ]
    est = Pipeline(steps)

    return random_grid, param_grid, est

def lsvcTune(rnd_state):
    # {'svc__kernel': 'linear', 'svc__gamma': 0.001, 'svc__C': 1, 'pca__n_components': 90}
    random_grid = {
        'svc__C': [1, 10, 100, 1000],
        'pca__n_components': [10,30,50,70,90,110,130]
        }

    param_grid= {
        'svc__C': [0.1,0.5,0.005],
        'pca__n_components': [9,10,11]
        }

    steps = [
        ('sscl', StandardScaler()),
        ('pca', PCA(random_state=rnd_state)),
        ('svc', SVC(random_state=rnd_state))
    ]
    est = Pipeline(steps)

    return random_grid, param_grid, est

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if not argm:
        quit()
    else:
        csv_file_name = argm[0]
        rand_flag = bool(int(argm[1]))

    # read data from the csv file
    file_reader = MSFileReader(csv_file_name)
    feature_data, labels, encoder = file_reader.encodeData()
    print(file_reader)

    training_features, training_labels = file_reader.getTICNormalization(), labels
    test_features, test_labels = file_reader.getTICNormalization(), labels


    """CREATE ESTIMATOR HERE"""
    rnd_state = None
    random_grid, param_grid, est = lsvcTune(rnd_state)

    if rand_flag:
        print('Performing Randomized Search')
        rf_random = RandomizedSearchCV(estimator = est, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, n_jobs = -1, random_state=rnd_state)# Fit the random search model
        rf_random.fit(training_features, training_labels)

        print(rf_random.best_params_)
    else:
        print('Performing Grid Search')
        grid_search = GridSearchCV(estimator = est, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(training_features, training_labels)

        print(grid_search.best_params_)

    

if __name__ == "__main__":
    main()
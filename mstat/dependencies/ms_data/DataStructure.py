from math import pi
from mstat.dependencies.ScikitImports import *

import joblib
import numpy as np
from matplotlib import projections, pyplot as plt
import sys, os
from datetime import *

from mstat.dependencies.ms_data.MSFileReader import MSFileReader

help_message = """
Console Command: python DataStructure.py <path/file_name.csv> <save_models>
Arguments:
    <path/file_name.csv>   - (String) path and name of CSV file including the extension ".csv"
    <>    - (String) 
    <> - (Boolean)"""

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def constructTrainTest(file_data, file_labels, option=0, tt_split=0.1, rand_state=0):
    np.random.seed(rand_state)
    # OPTION 0: Normal train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(
        file_data, file_labels, test_size=tt_split, stratify=file_labels, random_state=rand_state)
    
    if option == 1:
        # OPTION 1: Normal train-test split with added random noise points in test set
        # create random testing data
        rand_labels = np.array([-1] * int(0.25 * len(test_labels)))
        emp_cov = EmpiricalCovariance().fit(test_data)
        μ = emp_cov.location_
        Σ = emp_cov.covariance_

        rand_data = np.random.multivariate_normal(μ, 2 * Σ, rand_labels.shape[0])
        
        test_data = np.concatenate((test_data, rand_data))
        test_labels = np.concatenate((test_labels, rand_labels))
        
    elif option == 100:
        # OPTION 2: Normal train-test split with added random noise points in training set
        # create random training data
        rand_labels = np.array(["Unknown"] * len(test_labels))
        emp_cov = EmpiricalCovariance().fit(train_data)
        μ = emp_cov.location_
        Σ = emp_cov.covariance_

        rand_data = np.random.multivariate_normal(μ, 2 * Σ, rand_labels.shape[0])
        
        train_data = np.concatenate((train_data, rand_data))
        train_labels = np.concatenate((train_labels, rand_labels))

    elif option == 2:
        # OPTION 3: Leave half of classes (minimum of four training classes) out of the training set and include them in test data as structured/clustered unknowns
        shuffled_classes = np.unique(train_labels)
        np.random.shuffle(shuffled_classes)
        if len(shuffled_classes[:-2]) < 4:
            known, unknown = shuffled_classes[:4], shuffled_classes[4:]
        else:
            known, unknown = shuffled_classes[:-4], shuffled_classes[-4:]

        print(f"Unknown classes: {unknown}")

        known_ind = np.isin(file_labels, known)
        train_data, test_data_known, train_labels, test_labels_known = train_test_split(
            file_data[known_ind], file_labels[known_ind], test_size=tt_split, stratify=file_labels[known_ind], random_state=rand_state)
        test_data_unknown, test_labels_unknown = file_data[~known_ind], file_labels[~known_ind]

        test_data, test_labels = np.concatenate((test_data_known, test_data_unknown)), np.concatenate((test_labels_known, test_labels_unknown))

    return train_data, test_data, train_labels, test_labels

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if not argm:
        quit()
    else:
        file_name = argm[0]
        option = 1
        tt_split = 0.2
        rand_state = 13
        
    # read file data from the csv file
    file_reader = MSFileReader(file_name)
    _, _, file_labels, _ = file_reader.encodeData()
    print(file_reader)

    # create training and testing sets
    train_data, test_data, train_labels, test_labels = constructTrainTest(file_reader.getTICNormalization(), file_labels, option, tt_split, rand_state)


if __name__ == "__main__":
    main()
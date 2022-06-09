# coding: utf-8
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt, cm
import colorcet as cc
import sys
from datetime import *

from dependencies.ms_data.MSFileReader import MSFileReader
from dependencies.readModelConfig import *

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

help_message = """
Console Command: python AnalyseCorrelation.py <path/base_file_name.csv> <path/test_file_name.csv>
Arguments:
    <path/base_file_name.csv>    - (String) path and name of base data CSV file including the extension ".csv" 
    <path/test_file_name.csv>    - (String) path and name of test data CSV file including the extension ".csv" """

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

preprocessing_options = ['none', 'sscl', 'rscl', 'ptfm']

def main():
    argm = handleStartUpCommands(help_message)
    if not argm:
        print("Type 'python AnalyseCorr.py help' for more info")
        quit()
    else:
        file_name = argm[0]
        test_file_name = argm[1]

    # read data from the csv file
    file_reader = MSFileReader(file_name)
    _, _, labels, encoder = file_reader.encodeData()
    print(' LIBRARY DATA FROM CSV FILE '.center(80, '*'))
    print(file_reader)

    # read test data from the csv file
    test_reader = MSFileReader(test_file_name)
    frame, data, test_labels, t_encoder = test_reader.encodeData()
    print(' TEST DATA FROM CSV FILE '.center(80, '*'))
    print(test_reader)

    lib_data = file_reader.getTICNormalization()
    class_names = np.unique(labels)
    u_labels = class_names
    scl = StandardScaler()
    lib_data = scl.fit_transform(lib_data)

    #feature_select = VarianceThreshold(0.000005)
    #lib_data = feature_select.fit_transform(lib_data)

    #print(lib_data.shape)
    
    test_data = test_reader.getTICNormalization()
    test_data = scl.transform(test_data)
    #test_data = feature_select.transform(test_data)
    test_ind = list(range(len(test_labels)))

    clm = cm.get_cmap('cet_glasbey_light') 
    
    np.set_printoptions(precision=2, suppress=True)
    fracs = []
    obs_classes = []
    for ind in test_ind:
        print(f' Sample {ind+1} '.center(80, '*'))
        print(f"Looking at data from the {test_labels[ind]} class.")
        obs_classes.append(test_labels[ind])
        sample = test_data[ind]

        scores = []
        totals = []
        printProgressBar(0, len(u_labels), prefix = 'Progress:', suffix = 'Complete', length = 50)
        for num, label in enumerate(u_labels):
            # get data for current label
            X = lib_data[(labels == label), :]

            # get correlation scores for library data
            CORR = np.corrcoef(X)

            # compare correlation scores to test sample
            score = 0
            total = 0
            for i in range(1, X.shape[0]):
                # find correlation score for test sample with current library sample i
                CORR_sample_mat = np.corrcoef(X[i, :], sample)
                sample_corr = CORR_sample_mat[0, 1] 
                for j in range(i):
                    # keep track of how well the test sample correlates to library
                    score += int(CORR[i, j] <= sample_corr)
                    total += 1

                
            scores.append(score)
            totals.append(total)
            printProgressBar(num+1, len(u_labels), prefix = 'Progress:', suffix = 'Complete', length = 50)

        #print(scores)
        #print(totals)
        fracs.append(np.divide(scores, totals))
        print(f"Class predicted: {class_names[np.divide(scores, totals).argmax()]}\n")

    fig, ax = plt.subplots(figsize=(12, 8))
    
    wdth = 0.2

    #print(fracs)
    u_labels = encoder.transform(u_labels)
    for i, frac in enumerate(fracs):
        ax.bar(u_labels + i*wdth, frac, color = clm(i), width = wdth)
    
    ax.set_title(f'Samples {test_ind}')
    ax.set_xticks(u_labels + (len(fracs)-1)*wdth/2)
    ax.set_xticklabels(class_names, rotation = -90)
        #ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
        #ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
    ax.legend(obs_classes)
    plt.show()



        

if __name__ == "__main__":
    main()
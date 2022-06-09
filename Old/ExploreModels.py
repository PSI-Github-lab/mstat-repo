# coding: utf-8
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib import projections, pyplot as plt
import sys, os
from datetime import *

from dependencies.ms_data.MSFileReader import MSFileReader
from dependencies.ms_data.AnalysisVis import AnalysisVis
from dependencies.ms_data.MSDataAnalyser import MSDataAnalyser
from readModelConfig import *

def plotConfusionMatrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap='coolwarm')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, "%0.2f" % cm[i, j],
                        ha="center", va="center", color="w")

    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True label')

    fig.tight_layout()

def plotModelScores(results, names):
    training_means, training_stds = [ sub['training_mean'] for sub in results ], [ sub['training_std'] for sub in results ]
    scores = [ sub['score'] for sub in results ]

    ind = np.arange(len(training_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,10))
    rects1 = ax.bar(ind - width/2, training_means, width, yerr=training_stds, capsize=5,
                    label='Training/CV')
    rects2 = ax.bar(ind + width/2, scores, width, capsize=5,
                    label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance')
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    ax.grid()
    ax.legend()
    
help_message = """
Console Command: python testModels.py <path/file_name.csv> <path/model_configs.txt> <save_models>
Arguments:
    <path/training_name.csv> - (String) path and name of CSV file including the extension ".csv"
    <path/test_name.csv> - (String) path and name of CSV file including the extension ".csv"
    <path/model_configs.txt>  - (String)
    <plot_accuracy> - (Boolean)"""

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if not argm:
        quit()
    else:
        train_file_name = argm[0]
        test_file_name = argm[1]
        config_file_name = argm[2]
        plot_conf_flag = bool(int(argm[3]))

    # read training/crossval data from the csv file
    training_reader = MSFileReader(train_file_name)
    _, training_labels, encoder = training_reader.encodeData()

    # read test data from the csv file
    test_reader = MSFileReader(test_file_name)
    test_data, test_labels, _ = test_reader.encodeData()

    print(' TRAINING AND TESTING DATA '.center(80, '*'))
    print(training_reader)
    print(test_reader)

    # read model configurations from config file
    configs, file_lines = readModelConfig(config_file_name)

    # perform model analyses
    rnd_state = 42
    results = []
    names = []
    for config, file_line in zip(configs, file_lines):
        # create the estimator from line in config file and fit to training data
        pipeline = createModelPipeline(config, rnd_state)
        print(f' TESTING MODEL {file_line} '.center(80, '*'))
        print(pipeline)

        analysis = MSDataAnalyser(pipeline)
        analysis.fitModel(training_reader.getTICNormalization(), training_labels)
        class_names = encoder.inverse_transform(np.unique(training_labels))

        # perform cross validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd_state)
        cv_results = analysis.crossvalModel(class_names, cv)

        # perform test
        report, confusion = analysis.testModel(test_reader.getTICNormalization(), test_labels, class_names)
        test_score = analysis.pipeline.score(test_reader.getTICNormalization(), test_labels)
        print(report)

        #result_summary = f"[{file_line}]" + " Model has cross validation accuracy of %0.3f +/- %0.2f and a test accuracy of %0.3f" % (cv_results['test_score'].mean(), cv_results['test_score'].std(), test_score)
        #results.append(result_summary)

        results.append(dict(training_mean= cv_results['test_score'].mean(), training_std= cv_results['test_score'].std(), score= test_score))
        names.append(file_line)

        print("""   Avg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean()))
        #if plot_conf_flag:
        #    plotConfusionMatrix(confusion, class_names, f"[{file_line}]")
    
    if plot_conf_flag:
        plotModelScores(results, names)
        plt.show()

    print(' EXPLORATION RESULTS '.center(80, '*'))
    for result in results:
        print(result)

    

if __name__ == "__main__":
    main()
# coding: utf-8
try:
    from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
    from sklearn_hierarchical_classification.constants import ROOT
    from sklearn.calibration import CalibratedClassifierCV as CalClass
    from sklearn.model_selection import cross_validate
    from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx
    from pyswarms.utils.plotters import plot_cost_history, plot_contour
    import numpy as np
    from matplotlib import pyplot as plt, cm
    import colorcet as cc
    import sys
    from datetime import *
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
    from mstat.dependencies.ms_data.DataStructure import constructTrainTest
    from mstat.dependencies.readModelConfig import *
    from mstat.dependencies.hier_clustering import getHier, prob_dendrogram
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

training_data = None
training_labels = None

def set_global_data(data, labels):
    global training_data
    training_data = data
    global training_labels 
    training_labels = labels

def model_obj_func(x):
    if x.shape[1] != 1:
        raise IndexError("objective function only takes one-dimensional input.")
    
    x_ = np.ceil(x[:, 0])
    #print('x_', x_)

    results = np.ones(x_.shape)
    for i, n_comp in enumerate(x_):
        steps1 = [
            #('scl', StandardScaler()),
            ('dim', PCA(n_components=int(n_comp))),
            ('lda', LDA())
            #('svc', SVC(kernel='rbf', C=1, probability=True, random_state=rnd_state)),
        ]
        pred_est = Pipeline(steps1)

        # perform cross validation
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        cv_results = cross_validate(pred_est, training_data, training_labels, cv=cv, scoring='balanced_accuracy')
        results[i] = 1 - cv_results['test_score'].mean()

    return results


help_message = """
Console Command: python AnalyseCorrelation.py <path/base_file_name.csv> <path/test_file_name.csv>
Arguments:
    <path/base_file_name.csv>    - (String) path and name of base data CSV file including the extension ".csv" 
    <path/test_file_name.csv>    - (String) path and name of test data CSV file including the extension ".csv"
    <file_number_row>            - (Int)    choose row/file from test csv to test with the model """


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
        print("Type 'python model_HierPCALDA.py help' for more info")
        quit()
    else:
        file_name = argm[0]
        test_file_name = argm[1]
        file_row = int(argm[2])
        rnd_state = 45

    # read training data from the csv file
    file_reader = MSFileReader(file_name)
    _, feature_data, training_labels, _ = file_reader.encodeData()
    print(file_reader)

    # read test data from the csv file or partition first file into train and testing set
    if test_file_name.count('.csv') > 0:
        test_reader = MSFileReader(test_file_name)
        _, _, test_labels, _ = test_reader.encodeData()
        print(test_reader)

        training_data = file_reader.getTICNormalization()
        test_data = test_reader.getTICNormalization()
    else:
        training_data = file_reader.getTICNormalization()
        print(f'Training set {training_data.shape}')

    encoder = LabelEncoder().fit(np.unique(training_labels))
    n = len(encoder.classes_)
    X = np.empty((n, training_data.shape[1]))

    for i, label in enumerate(np.unique(training_labels)):
        class_data = training_data[(training_labels == label)]

        mean = np.mean(class_data, axis=0)
        X[i] = mean
    print(X.shape)

    # Set-up hyperparameters
    options = {'c1': 0.8, 'c2': 0.5, 'w': 0.9}

    # Call instance of PSO
    numb_particles = 10
    optimizer = ps.single.GlobalBestPSO(n_particles=numb_particles, dimensions=1, options=options, bounds=(np.array([3]), np.array([1000])), init_pos=np.linspace(10, 200, numb_particles).reshape(-1,1))

    set_global_data(training_data, training_labels)

    # Perform optimization
    best_cost, best_pos = optimizer.optimize(model_obj_func, iters=10)
    plot_cost_history(optimizer.cost_history)
    plt.show()

    steps1 = [
        #('scl', StandardScaler()),
        ('dim', PCA(random_state=rnd_state, n_components=int(best_pos))),
        ('lda', LDA())
        #('svc', SVC(kernel='rbf', C=1, probability=True, random_state=rnd_state)),
    ]
    pred_est = Pipeline(steps1)

    

    # perform cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd_state)
    cv_results = cross_validate(pred_est, training_data, training_labels, cv=cv, scoring='balanced_accuracy')
    print(" Model has cross validation accuracy of %0.3f +/- %0.3f" % (cv_results['test_score'].mean(), cv_results['test_score'].std()))
    print("""   Avg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean()))

    

if __name__ == "__main__":
    main()
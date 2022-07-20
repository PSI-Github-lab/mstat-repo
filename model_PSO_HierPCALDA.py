# coding: utf-8
try:
    from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
    from sklearn_hierarchical_classification.constants import ROOT
    from sklearn.calibration import CalibratedClassifierCV as CalClass
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import precision_score, make_scorer
    from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx
    from pyswarms.utils.plotters import plot_cost_history, plot_contour
    import numpy as np
    from matplotlib import pyplot as plt, cm
    from matplotlib import animation
    from PIL import Image
    import colorcet as cc
    import sys
    from datetime import *
    import time
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
hdict = None

def set_global_data(data, labels):
    global training_data
    training_data = data
    global training_labels 
    training_labels = labels
    global hdict
    hdict = class_hierarchy = {
        '<ROOT>': ['SKIN', 'Heart', 'Lung', 'MELANOMA', 'B'],
        'B': ['D458', 'D341', 'AMEL MELANOMA', 'UW228', 'ONS76', 'MDA', 'LM2-4']
    }

def model_obj_func(x):
    if x.shape[1] != 2:
        raise IndexError("objective function only takes two-dimensional input.")
    
    x_ = x[:, 0]
    y_ = x[:, 1]
    print('\nposn x', x_)
    print('posn y', y_)

    results = np.ones(x_.shape)
    for i, (n_comp1, n_comp2) in enumerate(zip(x_, y_)):
        steps1 = [
            ('dim', TruncatedSVD(n_components=int(n_comp1))),
            ('lda', LDA())
        ]
        root_pred = Pipeline(steps1)

        steps2 = [
            ('dim', TruncatedSVD(n_components=int(n_comp2))),
            ('lda', LDA())
        ]
        b_pred = Pipeline(steps2)

        hier_est = HierarchicalClassifier(
        base_estimator={
            '<ROOT>' : root_pred,
            'B' : b_pred,
        },
        class_hierarchy=hdict,
        )

        # perform cross validation
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        scorer = make_scorer(precision_score, average = 'weighted', zero_division=1)
        cv_results = cross_validate(hier_est, training_data, training_labels, cv=cv, scoring=scorer)
        error_rate = 1 - cv_results['test_score'].mean()
        dim1_penalty = np.tanh(n_comp1 / training_data.shape[1])
        dim2_penalty = np.tanh(n_comp2 / training_data.shape[1])
        results[i] = error_rate

    print('result', results)
    print('avgres', np.mean(results))
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
    options = {'c1': 0.9, 'c2': 0.1, 'w': 0.9}
    r_min, r_max = 10, 500

    # Call instance of PSO
    numb_particles = 20
    numb_iterations = 20
    numb_dimensions = 2
    optimizer = ps.single.GlobalBestPSO(
        n_particles=numb_particles, 
        dimensions=numb_dimensions, 
        options=options, 
        bounds=(np.array([3, 3]), np.array([1000, 1000])), 
        init_pos=np.random.randint(low=10, high=500, size=(numb_particles, numb_dimensions)).astype(float)
        )

    set_global_data(training_data, training_labels)

    # Perform optimization
    start_time = time.time()
    best_cost, best_pos = optimizer.optimize(model_obj_func, iters=numb_iterations)
    opt_time = time.time() - start_time
    if opt_time < 120:
        print(f"--- optimization completed in {(time.time() - start_time)} seconds ---")
    else:
        print(f"--- optimization completed in {(time.time() - start_time) / 60} minutes ---")
    
    plot_cost_history(optimizer.cost_history)
    
    # Make animation
    fig, ax = plt.subplots()

    x_lims = (r_min, r_max)
    y_lims = (r_min, r_max)

    animation = plot_contour(
        pos_history=optimizer.pos_history,
        canvas=(fig, ax),
        #mark=(1.3494066,  1.34940664),
        )
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_title('PSO Model Optimization')
    ax.set_xlabel('# dimensions for PCA 1')
    ax.set_ylabel('# dimensions for PCA 2')
    animation.save('plot0.gif', writer='imagemagick', fps=2)
    #Image(url='plot0.gif')
    plt.show()

    

if __name__ == "__main__":
    main()
# coding: utf-8
try:
    from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
    from sklearn_hierarchical_classification.constants import ROOT
    from sklearn.calibration import CalibratedClassifierCV as CalClass
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    import numpy as np
    from matplotlib import pyplot as plt, cm
    import colorcet as cc
    import sys
    from datetime import *
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
    from mstat.dependencies.ms_data.DataStructure import constructTrainTest
    from mstat.dependencies.readModelConfig import *
    from model_HierarchyClustering import getHier, prob_dendrogram
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

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
        print("Type 'python AnalyseCorr.py help' for more info")
        quit()
    else:
        file_name = argm[0]
        test_file_name = argm[1]
        file_row = int(argm[2])
        rnd_state = 44

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
        training_data, test_data, training_labels, test_labels = constructTrainTest(
            file_reader.getTICNormalization(), training_labels, option=int(test_file_name), tt_split=0.1, rand_state=rnd_state)

        print(f"\nNo test file given. Generated training set of length {len(training_data)} and testing set of length {len(test_data)}.\n")

    encoder = LabelEncoder().fit(np.unique(training_labels))
    n = len(encoder.classes_)
    X = np.empty((n, training_data.shape[1]))

    for i, label in enumerate(np.unique(training_labels)):
        class_data = training_data[(training_labels == label)]

        mean = np.mean(class_data, axis=0)
        X[i] = mean
    print(X.shape)

    hdict, linkage_matrix = getHier(X, encoder)

    copy = linkage_matrix
    copy[:,2] = np.array(np.arange(1, n))
    
    '''class_hierarchy = {
        ROOT: ["Fish", "Meat", "Liver", "Skin", 'MouseBrain', 'MouseKidney'],
        "Fish": ['RainbowTrout', 'SteelheadTrout', 'BasaFillet', 'TunaFillet'],
        "Meat": ['Pork', 'Beef', 'MouseMuscle', 'ChickenMeat'],
        "Liver": ['ChickenLiver', 'MouseLiver'],
        "Skin": ['ChickenSkin', 'PigSkin'],
    }'''
  

    params = {'dim__n_components': 100}
    steps = [
        #('scl', StandardScaler()),
        #('dim', TruncatedSVD()),
        ('svc', SVC(kernel='rbf', C=1, probability=True,
                          random_state=rnd_state))
    ]
    base_est = Pipeline(steps)
    #base_est.set_params(**params)
    #cal_est = CalClass(base_estimator=base_est, cv=2)

    hier_est = HierarchicalClassifier(
        base_estimator=base_est,
        class_hierarchy=hdict,
        #prediction_depth='nmlnp',
        #stopping_criteria=0.9
        #algorithm='lcn',
        #training_strategy='inclusive'

    )

    

    print(hier_est)
    hier_est.fit(training_data, training_labels)

    #test_data = feature_select.transform(test_data)
    test_ind = [file_row]
    print(test_labels[test_ind])

    probs = hier_est.predict_proba(list( test_data[i] for i in test_ind ))
    preds = hier_est.predict(list( test_data[i] for i in test_ind ))
    print(probs)
    print(preds)

    class_order = prob_dendrogram(np.concatenate((np.array([-1.0]), probs[0])), copy, truncate_mode="level", p=n-1, labels=encoder.classes_, leaf_rotation=90, color_threshold=0.0)
    
    # show final probability for each class at bottom of plot
    leaves = hier_est.classes_[-n:]
    print(leaves)
    leaf_probs = probs[0][-n:]
    print(leaf_probs)
    prob_colormap = cm.get_cmap('brg')
    for i, class_ in enumerate(leaves):
        prob = leaf_probs[i]
        gind = class_order.index(encoder.transform([class_])[0])
        x = 10*gind + 5
        plt.plot(x, 0, 'o', c=prob_colormap(0.4 + prob/2), markersize=75*(0.5 + prob/2))
        plt.annotate("%.2f" % prob, (x, 0), xytext=(0, 12),
                        textcoords='offset points',
                        va='top', ha='center', color='white', fontsize=12, weight='bold' if prob > 0.5 else 'normal')
    
    plt.show()

    '''
    clm = cm.get_cmap('cet_glasbey_light') 
    np.set_printoptions(precision=3, suppress=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    wdth = 0.5
    ind = np.arange(len(pipeline.classes_))
    for i, prob in enumerate(probs):
        ax.bar(ind, prob, color = clm(i), width=wdth)
    
    ax.set_title(f'Samples {test_ind}')
    ax.set_xticks(ind)
    ax.set_xticklabels(pipeline.classes_, rotation = -90)
    ax.legend(test_ind)
    plt.show()'''

if __name__ == "__main__":
    main()
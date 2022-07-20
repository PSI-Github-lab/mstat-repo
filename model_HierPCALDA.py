# coding: utf-8
try:
    from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
    from sklearn_hierarchical_classification.constants import ROOT
    from sklearn.calibration import CalibratedClassifierCV as CalClass
    from sklearn.model_selection import cross_validate
    from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA
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
    from mstat.dependencies.hier_clustering import getHier, prob_dendrogram
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()


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

    '''hdict, linkage_matrix = getHier(X, encoder)
    print(hdict, linkage_matrix)

    copy = linkage_matrix
    copy[:,2] = np.array(np.arange(1, n))'''

    '''class_hierarchy = {
        ROOT: ["Fish", "Meat", "Liver", "Skin", 'MouseBrain', 'MouseKidney'],
        "Fish": ['RainbowTrout', 'SteelheadTrout', 'BasaFillet', 'TunaFillet'],
        "Meat": ['Pork', 'Beef', 'MouseMuscle', 'ChickenMeat'],
        "Liver": ['ChickenLiver', 'MouseLiver'],
        "Skin": ['ChickenSkin', 'PigSkin'],
    }'''

    class_hierarchy = {'B': ['C', 'Epend PostA', 'Epend PostB'], 'C': ['Epend YAP1', 'D'], 'D': ['Epend RELA', 'Med WNT', 'Med SHH', 'Med Group 3', 'Med Group 4'], '<ROOT>': ['Pilo Astro', 'B']}
  
    steps1 = [
        #('scl', StandardScaler()),
        ('dim', TruncatedSVD(random_state=rnd_state, n_components=40)),
        ('lda', LDA())
        #('svc', SVC(kernel='rbf', C=1, probability=True, random_state=rnd_state)),
    ]
    base1 = Pipeline(steps1)

    steps2 = [
        ('dim', TruncatedSVD(random_state=rnd_state, n_components=20)),
        ('lda', LDA())
    ]
    base2 = Pipeline(steps2)

    steps3 = [
        ('dim', TruncatedSVD(random_state=rnd_state, n_components=10)),
        ('lda', LDA())
    ]
    base3 = Pipeline(steps3)

    hier_est = HierarchicalClassifier(
        base_estimator={
            '<ROOT>' : base3,
            'B' : base2,
            'C' : base3,
            'D' : base1
        },
        class_hierarchy=class_hierarchy,
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

    # perform cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd_state)
    cv_results = cross_validate(hier_est, training_data, training_labels, cv=cv, scoring='balanced_accuracy')
    print(" Model has cross validation accuracy of %0.3f +/- %0.3f" % (cv_results['test_score'].mean(), cv_results['test_score'].std()))
    print("""   Avg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean()))

    '''class_order = prob_dendrogram(np.concatenate((np.array([-1.0]), probs[0])), copy, truncate_mode="level", p=n-1, labels=encoder.classes_, leaf_rotation=90, color_threshold=0.0)
    
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
    
    plt.show()'''

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
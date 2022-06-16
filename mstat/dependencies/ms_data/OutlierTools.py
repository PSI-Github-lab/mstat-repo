# coding: utf-8
try:
    import numpy as np
    from matplotlib import projections, pyplot as plt
    import sys, os
    from datetime import *
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
    from mstat.dependencies.ScikitImports import *
except ModuleNotFoundError as e:
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

'''Feature significance with ANOVA https://datascience.stackexchange.com/questions/74465/how-to-understand-anova-f-for-feature-selection-in-python-sklearn-selectkbest-w'''

def removeOutliers(training_data, training_labels, verbose=False, rnd_state=42):
    pipeline = Pipeline(
        [
            ('scl', StandardScaler()),
            #('dim', PCA(n_components=30, random_state=rnd_state)),
            ('lof', LocalOutlierFactor(n_neighbors=20)),
            #('ell', EllipticEnvelope(random_state=rnd_state)),
        ]
    )

    total_outliers = np.array([])
    new_data = np.array([])
    new_labels = np.array([])

    for label in np.unique(training_labels):
        class_data = training_data[(training_labels == label)]

        outliers = pipeline.fit_predict(class_data)

        if verbose:
            print(f"{label} data\nLength: {class_data.shape[0]}\t# outliers: {sum((outliers == -1))}")

        try:  
            new_data = np.concatenate((new_data, class_data[(outliers == 1)][0]))
            new_data = np.vstack((new_data, class_data[(outliers == 1)][1:]))
        except:
            new_data = np.vstack((new_data, class_data[(outliers == 1)]))
        new_labels = np.concatenate((new_labels, np.array(sum((outliers == 1)) * [label])))

        total_outliers = np.concatenate((total_outliers, outliers))
        #print(new_data.shape, new_labels.shape)
    
    if verbose:
        print(f"All data\nLength: {training_data.shape[0]}\t# outliers: {sum((total_outliers == -1))}")

    return new_data, new_labels
    
help_message = """
Console Command: python testModel.py <path/file_name.csv> <save_models>
Arguments:
    <path/train_name.csv>   - (String) path and name of CSV file including the extension ".csv"
    <path/test_name.csv>    - (String) path and name of CSV file including the extension ".csv"
    <plot_confusion_matrix> - (Boolean)"""

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
    # read training data from the csv file
    training_reader = MSFileReader(train_file_name)
    _, _, training_labels, encoder = training_reader.encodeData()
    print(training_reader)

    # read test data from the csv file
    if test_file_name.count('.csv') > 0:
        test_reader = MSFileReader(test_file_name)
        _, _, test_labels, test_encoder = test_reader.encodeData()
        print(test_reader)

        training_data = training_reader.getTICNormalization()
        
        test_data = test_reader.getTICNormalization()

    else:
        training_data, test_data, training_labels, test_labels = train_test_split(
            training_reader.getTICNormalization(), training_labels, test_size=0.2, stratify=training_labels)

        print(f"\nNo test file given. Generated training set of length {len(training_data)} and testing set of length {len(test_data)}.\n")


    """CREATE MODEL HERE"""
    rnd_state = None

    clean_data, clean_labels = removeOutliers(training_data, training_labels, encoder)
    

if __name__ == "__main__":
    main()
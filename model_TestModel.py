# coding: utf-8
try:
    import sys, os
    import bisect
    import numpy as np
    from numpy.lib.function_base import average, median
    from matplotlib import pyplot as plt
    import re
    from mstat.dependencies.ScikitImports import *
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
    from mstat.dependencies.ms_data.DataStructure import constructTrainTest
    from mstat.dependencies.ms_data.OutlierTools import removeOutliers
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def plotConfusionMatrix(cm, labels, title):
    """ Plot confusion matrix given a pre-calcuated matrix, class labels, and an appropriate title"""
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
    
help_message = """
Console Command: python TestModel.py python TestModel.py <train file.csv> <test file data mode> <pca dimension>
<confidence threshold> <plot confusion matrix> <model file>
Arguments:
    <path/train_file>          - (String) path and name of CSV file including the extension ".csv"
    <path/test_file_data_mode> - (String) path and name of CSV file including the extension ".csv"
                                 OR
                                 (Integer) 0 - generate random test group from first file
                                           1 - generate random test group from first file + random noise points
                                           2 - generate random test group from first file + held-out classes 
    <pca_dimension>            - (Integer) number of PCA dimensions 
    <confidence_threshold>     - (Float) confidence level to decide between class prediction and unknown sample
    <plot_confusion_matrix>    - (Boolean) select whether or not to plot a confusion matrix
    <path/model_file>         - (String) model file including extension ".model"
                                 OR
                                 "save" which creates a new model file """

def handleStartUpCommands(help_message):
    """ extract individual arguments from command line input
        then return arguments in an array"""
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def predict_unknown(pred_est, test_data, test_labels, outlier_detect=None, alpha=0.9, verbose=False, unknown_label='unknown'):
    """ Predict whether data points belong to the distribution which trained model is based on
        using an outlier detection and probability threshold scheme """
    new_labels = np.copy(test_labels)

    probs = pred_est.predict_proba(test_data)
    preds = pred_est.predict(test_data)
    if outlier_detect is not None:
        outl = outlier_detect.predict(test_data)
        #print(outl)
    else:
        outl = np.array([1] * len(preds))

    for i in range(len(probs)):
        #print(probs[i], max(probs[i]))
        if new_labels[i] not in pred_est.classes_:
            new_labels[i] = unknown_label
        if outl[i] == -1:
            preds[i] = unknown_label
        if max(probs[i]) < alpha:
            preds[i] = unknown_label
        if ((preds[i] != unknown_label and new_labels[i] == unknown_label) or (preds[i] == unknown_label and new_labels[i] != unknown_label)) and verbose: 
            print(test_labels[i], probs[i], preds[i])
        
        if unknown_label in preds:
            class_names = np.concatenate((pred_est.classes_, np.array([unknown_label])))
        else:
            class_names = pred_est.classes_

    return new_labels, preds, class_names

def main():
    if argm := handleStartUpCommands(help_message):
        train_file_name = argm[0]
        test_file_name = argm[1]
        pca_dim = int(argm[2])
        conf_thresh = float(argm[3])
        plot_conf_flag = bool(int(argm[4]))
        model_file = argm[5]
        rnd_state = 42

    else:
        quit()
    # read training data from the csv file
    if train_file_name.count('.csv') < 1:
        print("ERROR: Training file must be a csv")
        quit()
    training_reader = MSFileReader(train_file_name)
    _, _, training_labels, encoder = training_reader.encodeData()
    print(training_reader)

    # read test data from the csv file or partition first file into train and testing set
    if test_file_name.count('.csv') > 0:
        test_reader = MSFileReader(test_file_name)
        _, _, test_labels, _ = test_reader.encodeData()
        print(test_reader)

        training_data = training_reader.getTICNormalization()

        test_data = test_reader.getTICNormalization()
    else:
        training_data, test_data, training_labels, test_labels = constructTrainTest(
            training_reader.getTICNormalization(), training_labels, option=int(test_file_name), tt_split=0.2, rand_state=rnd_state)

        print(f"\nNo test file given. Generated training set of length {len(training_data)} and testing set of length {len(test_data)}.\n")

    # CREATE MODEL HERE
    if (model_file.count('.model') <= 0) or model_file == 'save':
        # if a model file is not specified, create and train model in this section
        steps = [
            #('scl', MinMaxScaler()),
            ('dim', PCA(n_components=pca_dim, random_state=rnd_state)),
            ('lda', LDA(store_covariance=True)),
            #('svc', SVC(kernel='rbf', C=1, probability=True,
            #              random_state=rnd_state))
            #('tre', RandomForestClassifier(random_state=rnd_state))
            ]
        base = Pipeline(steps)
        pred_est = base#CalClass(base_estimator=base, method='sigmoid', cv=3, ensemble=True)

        # get rid of outliers in training set
        #training_data, training_labels = removeOutliers(training_data, training_labels, rnd_state=rnd_state, verbose=False)

        # fit the model with training data
        pred_est.fit(training_data, training_labels)

        # create outlier estimator and train
        outl_est = Pipeline(
        [
        #('scl', PowerTransformer()),
        #('dim', PCA(n_components=pca_dim, random_state=rnd_state)),
        #('lda', LDA()),
        ('lof', LocalOutlierFactor(novelty=True, n_neighbors=20)),
        #('ocsvm', OneClassSVM(nu=0.1)),
        ]).fit(training_data, training_labels)
        outl_est = None

        if  model_file == 'save':
            # save model to external file
            # create file name based on the model structure
            r = re.compile(".*__")
            pred_params = list(pred_est.get_params().keys())
            pred_ind = pred_params.index(list(filter(r.match, pred_params))[0])
            pred_name = '-'.join(pred_params[3:pred_ind])

            if outl_est != None:
                #print(list(outl_est.get_params().keys())[3:])
                outl_params = list(outl_est.get_params().keys())
                outl_ind = outl_params.index(list(filter(r.match, outl_params))[0])
                outl_name = '-'.join(outl_params[3:outl_ind])

                name = f"{pred_name}_{outl_name}" 
            else:
                name = pred_name

            # meta data from model parameters and info about training file and random seed
            model_dict = pred_est.get_params()
            model_dict['training_file'] = train_file_name
            model_dict['random_seed'] = rnd_state
            meta_info = model_dict

            joblib.dump((pred_est, outl_est, meta_info), name + ".model")
            print(f"Generated and saved new model as {name}.model\n")
    else:
        # load pre-trained model from save file
        pred_est, outl_est, meta_info = joblib.load(model_file)
        print(f"Loaded model from {model_file}\n")

    print(pred_est)

    # transform test labels so all unknown data are properly recorded
    test_labels = np.array(['Unknown' if s not in encoder.classes_ else s for s in test_labels])
    le_classes = encoder.classes_.tolist()
    bisect.insort_left(le_classes, 'Unknown')
    if 'Unknown' in test_labels:
        encoder.classes_ = np.array(le_classes)

    # perform cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd_state)
    cv_results = cross_validate(pred_est, training_data, training_labels, cv=cv, scoring='balanced_accuracy')

    # check predicted probability distribution 
    like_pred = pred_est.predict_proba(test_data).max(axis=1)
    print(f"Maximum predicted likelihood:   {max(like_pred)}")
    print(f"Median predicted likelihood:    {median(like_pred)}")
    print(f"Minimum predicted likelihood:   {min(like_pred)}\n")
    #plt.hist(like_pred)

    # perform test accounting for unknowns
    test_labels, predicted, test_class_names = predict_unknown(pred_est, test_data, test_labels, alpha=conf_thresh, outlier_detect=outl_est) #how to choose α???
    if len(np.unique(predicted)) != len(np.unique(test_labels)):
        print(f"Model fails to predict every classes. Only see labels {np.unique(predicted)}\n")

    # create classification report
    report = classification_report(test_labels, predicted, zero_division=0, digits=3)
    confusion = confusion_matrix(test_labels, predicted, normalize='true')
    print(report)
    print(" Model has cross validation accuracy of %0.3f +/- %0.3f" % (cv_results['test_score'].mean(), cv_results['test_score'].std()))
    print("""   Avg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean()))

    # plot confusion matrix, if specified
    if plot_conf_flag:
        plotConfusionMatrix(confusion, test_class_names, 'Candidate Model')
    plt.show()

if __name__ == "__main__":
    main()

try:
    import joblib
    from mstat.dependencies.ScikitImports import *
    import pandas as pd
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    from PyQt5 import QtCore
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.ScikitImports import StratifiedKFold, cross_validate
    from mstat.dependencies.file_conversion.RAWConversion import raw_to_numpy_array
    from mstat.gui.main_gui import TestResultsGUI
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
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

def predict_unknown(pred_est, test_data, test_labels, outlier_detect=None, α=0.9, verbose=False, unknown_label='unknown'):
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

    # label unknown points
    for i in range(len(probs)):
        # check if the true label isn't from any of the trained classes
        if new_labels[i] not in pred_est.classes_:
            new_labels[i] = unknown_label
        # check if an outlier detector has labelled the point as an outlier (-1)
        if outl[i] == -1:
            preds[i] = unknown_label
        # if minimum probability threshold (α) isn't met for any class prediction then the point is labeled as an unknown
        if max(probs[i]) < α:
            preds[i] = unknown_label
        # extra confusing print line... 
        if ((preds[i] != unknown_label and new_labels[i] == unknown_label) or (preds[i] == unknown_label and new_labels[i] != unknown_label)) and verbose: 
            print(test_labels[i], probs[i], preds[i])
        
        # add the unknown label to the observed class names if any point has recieved that label
        if unknown_label in preds:
            class_names = np.concatenate((pred_est.classes_, np.array([unknown_label])))
        else:
            class_names = pred_est.classes_

    return new_labels, preds, probs, class_names

class TestWorkerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(CompFlag, int)
    progressComplete = QtCore.pyqtSignal(CompFlag, str, tuple)

class TestWorker(QtCore.QRunnable):
    def __init__(self, model : tuple, test_data : tuple, conf_thresh : float, plot_conf_flag=True, rnd_state=42):
        super(TestWorker, self).__init__()
        self.signals = TestWorkerSignals()
        self.pred_est, self.outl_est, self.meta_info = model
        self.test_features, self.test_labels = test_data
        self.conf_thresh = conf_thresh
        self.plot_conf_flag = plot_conf_flag
        self.rnd_state = rnd_state

    def run(self):
        # test model here
        print('TESTING MODEL')
        print(self.test_features.shape, self.test_labels.shape)
        
        results=""

        # perform test accounting for unknowns
        test_labels, predicted, _, test_class_names = predict_unknown(self.pred_est, self.test_features, self.test_labels, α=self.conf_thresh, outlier_detect=None)
        if len(np.unique(predicted)) != len(np.unique(test_labels)):
            results += f"Model fails to predict every classes. Only see labels {np.unique(predicted)}\n\n"
            print(results)
            
        report = classification_report(test_labels, predicted, zero_division=0, digits=3)
        results += report
        confusion = confusion_matrix(test_labels, predicted, normalize='true')
        print(report)

        self.signals.progressComplete.emit(CompFlag.SUCCESS, results, (confusion, test_class_names))

def on_test_worker_update(self, result, result2):
    pass

def on_test_worker_complete(self, comp_flag, results, conf_data):
    print(f"\tEnding worker...")
    self.num_processes -= 1

    if comp_flag == CompFlag.SUCCESS and self.num_processes == 0:
        self.progress_total = 0
        plot_conf_flag = True
        if plot_conf_flag:
            try:
                confusion, test_class_names = conf_data
                plotConfusionMatrix(confusion, test_class_names, 'Candidate Model')
                plt.show()
            except Exception as exc:
                print(f"Couldn't create confusion plot due to following error:\n\n{exc}")

        self.main_ctrl.test_results = TestResultsGUI(self.main_ctrl, results)
        self.main_ctrl.test_results.show()

        #self.main_ctrl.gui.showInfo("Model testing is complete!")
    else:
        self.main_ctrl.gui.showInfo("Model testing is completed with the following errors:\n\n")
        
    self.main_ctrl.set_state(ControlStates.READY)

def test_model(self):
    if not self.isTrained():
        self.main_ctrl.gui.showError("Model must be trained before testing data.")
        self.main_ctrl.set_state(ControlStates.READY)
    elif len(self.testing_dict) == 0:
        self.main_ctrl.gui.showError("Must select testing data to be tested.")
        self.main_ctrl.set_state(ControlStates.READY)
    else:
        print('MODEL PARAMETERS', self.pca_dim, self.bin_size, self.low_lim, self.up_lim)
        print('MODEL PARAMETERS (CONT)', self.do_diff, self.diff_order)

        _, test_features, test_labels = extract_feature_data(self.testing_dict, self.bin_size, self.low_lim, self.up_lim)
        test_data = (test_features, test_labels)
        conf_thresh = 0.0

        worker = TestWorker(self.model, test_data, conf_thresh)
        worker.setAutoDelete(True)
        worker.signals.progressChanged.connect(self.on_test_worker_update)
        worker.signals.progressComplete.connect(self.on_test_worker_complete)
        self.main_ctrl.threadpool.start(worker)
        #self.worker_progress_list.append(0)
        self.progress_total += 1
        self.num_processes += 1

def test_single_file(self):
    if not self.isTrained():
        self.main_gui.showError("Model must be trained before testing data.")
        self.main_ctrl.set_state(ControlStates.READY)
    else:
        file_name = self.main_gui.file_dialog(self.main_ctrl.main_testing_dir, dialog_caption='Select a file containing a sample', type_filter="RAW Files (*.raw);;NPY Files (*.npy)")

        if '.npy' in file_name.lower():
            with open(file_name, 'rb') as f:
                intens = np.load(f, allow_pickle=True)
                mzs = np.load(f, allow_pickle=True)
                meta = np.load(f, allow_pickle=True)

            if intens.shape[0] > 1:
                self.main_gui.showError(f"Must select file containing only one sample.\n\nThis file contains {intens.shape[0]} samples.")
            else:
                print(intens.shape, intens[0].shape, mzs.shape, meta)
                data = intens, mzs, meta
                self.on_data_worker_complete(CompFlag.SUCCESS, data, meta[0]['comment1'], file_name, DataRole.TESTING)
                
                conf_thresh = 0.0
                _, binned_data = bin_data(mzs[0], intens[0], None, self.bin_size, self.low_lim, self.up_lim)
                binned_data[np.isnan(binned_data)] = 0.
                new_labels, predicted, probs, _ = predict_unknown(self.pred_est, binned_data, [ meta[0]['comment1'] ], outlier_detect=None, α=conf_thresh)
                print('PREDICTIONS', new_labels, predicted, probs)
                self.main_gui.showInfo(f"{file_name}\n\nPredicted as {predicted[0]} with a confidence of {max(probs[0])}")

        elif '.raw' in file_name.lower():
            mzs, intens, meta = raw_to_numpy_array(file_name, sel_region=False, smoothing=False)
            print(intens.shape, meta)
            data = [intens], [mzs], [meta]
            self.on_data_worker_complete(CompFlag.SUCCESS, data, meta['comment1'], file_name, DataRole.TESTING)

            conf_thresh = 0.0
            _, binned_data = bin_data(mzs, intens, None, self.bin_size, self.low_lim, self.up_lim)
            binned_data[np.isnan(binned_data)] = 0.
            new_labels, predicted, probs, _ = predict_unknown(self.pred_est, binned_data.reshape(1, -1), [ meta['comment1'] ], outlier_detect=None, α=conf_thresh)
            print('PREDICTIONS', new_labels, predicted, probs)
            self.main_gui.showInfo(f"{file_name}\n\nPredicted as {predicted[0]} with a confidence of {max(probs[0])}")
        else:
            self.main_gui.showError("Must select an allowed file type for testing.")
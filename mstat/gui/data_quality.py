try:
    import os
    from os.path import exists
    import time
    import random
    from PyQt5 import QtCore, QtWidgets
    from sklearn.decomposition import PCA
    from scipy.stats import entropy
    from mstat.dependencies.helper_funcs import *
    from mstat.gui.mstatDataQualityDialogUI import Ui_Dialog as DataQualityDialogGUI
    from mstat.gui.main_gui import DataQualityResultsGUI
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class DataQualitySignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    progressComplete = QtCore.pyqtSignal(CompFlag, str, Exception)

class DataQualityWorker(QtCore.QRunnable):
    def __init__(self, classa_data):
        super(DataQualityWorker, self).__init__()
        self.signals = DataQualitySignals()
        self.classa_data = classa_data

    def run(self):
        time.sleep(1)

        classa_features, classa_labels = self.classa_data
        norm_data = getTICNormalization(classa_features)

        results = ''

        pca = PCA().fit(norm_data)
        percent_variance = pca.explained_variance_ratio_
        #print(percent_variance)
        threshold = 0.95

        for i in range(1, len(percent_variance)):
            if sum(percent_variance[:i]) >= threshold:
                #print(i, i / X_train.shape[0], sum(percent_variance[:i]))
                results += f"{threshold * 100}% variance captured with {i} principal components\n"
                results += f"PCA complexity score of {np.log(i / norm_data.shape[1])}\n\n"
                #print(percent_variance[:i])
                break

        avg_spectrum = np.mean(norm_data, axis=0)

        didm = np.gradient(avg_spectrum) #np.diff(avg_spectrum) / np.diff(bin_edges[:-1])
        diffRange = np.max( [ np.abs( didm.min() ), np.abs( didm.max() ) ] )

        num_bins = 1024
        hist_vals, _ = np.histogram(didm, bins=num_bins, range=[-diffRange, diffRange])

        results += f"Entropy, H = {0.5*entropy(hist_vals / sum(hist_vals), base=2)}" # / np.log(num_bins))

        self.signals.progressComplete.emit(CompFlag.SUCCESS, results, Exception())

class DataQualityCtrl(QtWidgets.QDialog):
    def __init__(self, main_ctrl, model_ctrl, main_gui):
        super().__init__()
        self.main_gui = main_gui
        self.model_ctrl = model_ctrl
        self.main_ctrl = main_ctrl

        self.dialog = DataQualityDialogGUI()
        self.dialog.setupUi(self)
        self.dialog.buttonBox.accepted.connect(self.dialog_accepted)
        self.dialog.buttonBox.rejected.connect(self.dialog_rejected)

        training_keys = self.model_ctrl.training_dict.keys()
        for key in training_keys:
            #print(key, self.model_ctrl.training_dict[key].keys())
            self.dialog.classa_combo.addItem(key)
        self.dialog.classa_combo.setCurrentIndex(0)

        self.show()

    def dialog_accepted(self):
        classa = self.dialog.classa_combo.currentText()
        classa_dirs = self.model_ctrl.training_dict[classa].keys()
        print(classa, classa_dirs)

        mzs, features, labels = self.model_ctrl.get_feature_data(DataRole.TRAINING)
        print('ALL DATA', features.shape)
        print(labels)

        classa_data = ( features[(labels == classa)], labels[(labels == classa)] )

        worker = DataQualityWorker(classa_data)
        worker.setAutoDelete(True)
        worker.signals.progressChanged.connect(self.on_data_worker_update)
        worker.signals.progressComplete.connect(self.on_data_worker_complete)
        
        self.main_ctrl.threadpool.start(worker) 

    def on_data_worker_complete(self, comp_flag, results, exc):
        print(f"\tEnding worker...")
        if comp_flag == CompFlag.SUCCESS:
            self.main_ctrl.data_quality_results = DataQualityResultsGUI(self.main_ctrl, results)
            self.main_ctrl.data_quality_results.show()
        else:
            self.main_gui.showError(f"Error with data quality analysis \n{self} \n\n{exc}")
        
    def on_data_worker_update(self, result):
        #self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setValue(result)
        #self.worker_progress_list[row_num] = result
        #print(f"\t{result} {row_num}")
        #self.main_gui.statusprogress_bar.setValue(sum(self.worker_progress_list))
        pass

    def dialog_rejected(self):
        self.close()
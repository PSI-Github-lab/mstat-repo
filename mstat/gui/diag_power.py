try:
    import os
    import time
    from os.path import exists
    import random
    from PyQt5 import QtCore, QtWidgets
    from sklearn.model_selection import LeaveOneOut
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.ms_data.DataMetrics import calcDK, calcPCAComplexity, calc1NNError
    from mstat.gui.mstatDiagPowerDialogUI import Ui_Dialog as DiagPowerDialogGUI
    from mstat.gui.main_gui import DiagPowerResultsGUI
    from tqdm import tqdm
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class DiagPowerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    progressComplete = QtCore.pyqtSignal(CompFlag, str, Exception)

class DiagPowerWorker(QtCore.QRunnable):
    def __init__(self, classa_data, classb_data):
        super(DiagPowerWorker, self).__init__()
        self.signals = DiagPowerSignals()

        self.classa_data = classa_data
        self.classb_data = classb_data

    def run(self):
        classa_features, classa_labels = self.classa_data 
        classb_features, classb_labels = self.classb_data

        norm_data = np.vstack((getTICNormalization(classa_features), getTICNormalization(classb_features)))
        labels = np.concatenate((classa_labels, classb_labels))

        DKC, SC, PC, NNE = self.datasetDP(norm_data, labels)

        results = f"name: {classa_labels[0]} shape: {classa_features.shape}\n\nname: {classb_labels[0]} shape: {classb_features.shape}"
        #results += f"\n\n{np.sum(np.sum(getTICNormalization(classa_features)))} {np.sum(np.sum(getTICNormalization(classb_features)))}"
        #results += f"\n\n{norm_data.shape} {labels.shape}"
        results += "\n\nLeaveOneOut CV Results\n"
        dkc_m, dkc_ci = mean_normal_cinterval(DKC, confidence=0.95)
        dkc_std = np.std(DKC, ddof=1)
        sc_m, sc_ci = mean_normal_cinterval(SC, confidence=0.95)
        sc_std = np.std(SC, ddof=1)
        textstr = 'DKC={:.3f}+-{:.3f}'.format(dkc_m, 2*dkc_std) + '\nSC={:.3f}+-{:.3f}\n'.format(sc_m, 2*sc_std)
        results += textstr
        pc_m, pc_ci = mean_normal_cinterval(PC, confidence=0.95)
        pc_std = np.std(PC, ddof=1)
        nne_m, nne_ci = mean_normal_cinterval(NNE, confidence=0.95)
        nne_std = np.std(NNE, ddof=1)
        textstr = 'PCA={:.3f}+-{:.3f}'.format(pc_m, 2*pc_std) + '\nNNE={:.3f}+-{:.3f}'.format(nne_m, 2*nne_std)
        results += textstr

        self.signals.progressComplete.emit(CompFlag.SUCCESS, results, Exception())

    def diagPower(self, feature_data, labels):
        my_metrics = []
        u_labels = np.unique(labels)
        #print(u_labels)
        # calculate fisher discriminant ratio
        #FDR = calcFDR(feature_data[(labels==u_labels[0]),:], feature_data[(labels==u_labels[1]),:])
        # calculate dong-kathari coef
        start_time = time.time()
        DKC = -1
        ind = -1
        for i in range(2, min(feature_data.shape[0], feature_data.shape[1])+1):
            data = PCA(n_components=i).fit_transform(feature_data)
            nDKC = calcDK(data[(labels==u_labels[0]),:], data[(labels==u_labels[1]),:])
            #print(nDKC)
            if nDKC > DKC:
                DKC = nDKC
                ind = i
            if DKC == 1.000:
                break
        my_metrics.append(DKC)
        #print(f"--- completed in {time.time() - start_time} seconds ---")

        # calculate silhouette_score
        start_time = time.time()
        SC = metrics.silhouette_score(feature_data, labels)
        my_metrics.append(SC)
        #print(f"--- completed in {time.time() - start_time} seconds ---")

        # calculate PCA complexity
        start_time = time.time()
        PC = calcPCAComplexity(feature_data)
        my_metrics.append(PC)
        #print(f"--- completed in {time.time() - start_time} seconds ---")

        # calculate 1NN performance
        start_time = time.time()
        NNE = 0.0#calc1NNError(feature_data, labels)
        my_metrics.append(NNE)
        #print(f"--- completed in {time.time() - start_time} seconds ---")

        return my_metrics

    def datasetDP(self, feature_data, labels):
        u_labels = np.unique(labels)
        DKCs = []
        SCs = []
        PCs = []
        NNEs = []
        
        loo = LeaveOneOut()
        n_splits = loo.get_n_splits(feature_data)

        for train_index, test_index in tqdm(loo.split(feature_data), desc="LOO DIAG POWER", total=n_splits):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = feature_data[train_index], feature_data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            D, S, P, N = self.diagPower(X_train, y_train)
            DKCs.append(D); SCs.append(S); PCs.append(P); NNEs.append(N)
        return DKCs, SCs, PCs, NNEs

class DiagPowerCtrl(QtWidgets.QDialog):
    def __init__(self, main_ctrl, model_ctrl, main_gui):
        super().__init__()
        self.main_gui = main_gui
        self.model_ctrl = model_ctrl
        self.main_ctrl = main_ctrl

        self.dialog = DiagPowerDialogGUI()
        self.dialog.setupUi(self)
        self.dialog.buttonBox.accepted.connect(self.dialog_accepted)
        self.dialog.buttonBox.rejected.connect(self.dialog_rejected)

        training_keys = self.model_ctrl.training_dict.keys()
        for key in training_keys:
            #print(key, self.model_ctrl.training_dict[key].keys())
            self.dialog.classa_combo.addItem(key)
            self.dialog.classb_combo.addItem(key)
        self.dialog.classa_combo.setCurrentIndex(0)
        self.dialog.classb_combo.setCurrentIndex(1)

        self.show()

    def dialog_accepted(self):
        #self.dialog.buttonBox.accepted.setEnabled(False)

        classa = self.dialog.classa_combo.currentText()
        classa_dirs = self.model_ctrl.training_dict[classa].keys()
        print(classa, classa_dirs)
        classb = self.dialog.classb_combo.currentText()
        classb_dirs = self.model_ctrl.training_dict[classb].keys()
        print(classb, classb_dirs)

        mzs, features, labels = self.model_ctrl.get_feature_data(DataRole.TRAINING)
        print('ALL DATA', features.shape)
        print(labels)

        classa_data = ( features[(labels == classa)], labels[(labels == classa)] )
        classb_data = ( features[(labels == classb)], labels[(labels == classb)] )

        worker = DiagPowerWorker(classa_data, classb_data)
        worker.setAutoDelete(True)
        worker.signals.progressChanged.connect(self.on_data_worker_update)
        worker.signals.progressComplete.connect(self.on_data_worker_complete)
        
        self.main_ctrl.threadpool.start(worker) 

    def on_data_worker_complete(self, comp_flag, results, exc):
        print(f"\tEnding worker...")
        if comp_flag == CompFlag.SUCCESS:
            self.main_ctrl.diag_power_results = DiagPowerResultsGUI(self.main_ctrl, results)
            self.main_ctrl.diag_power_results.show()
        else:
            self.main_gui.showError(f"Error with diagnostic power analysis \n{self} \n\n{exc}")
        
    def on_data_worker_update(self, result):
        #self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setValue(result)
        #self.worker_progress_list[row_num] = result
        #print(f"\t{result} {row_num}")
        #self.main_gui.statusprogress_bar.setValue(sum(self.worker_progress_list))
        pass

    def dialog_rejected(self):
        self.close()
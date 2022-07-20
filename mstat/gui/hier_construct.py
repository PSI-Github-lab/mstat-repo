try:
    import os
    import time
    import random
    from matplotlib import cm, pyplot as plt
    from PyQt5 import QtCore, QtWidgets
    from scipy.spatial import KDTree
    from mstat.dependencies.helper_funcs import *
    from sklearn.preprocessing import LabelEncoder
    from mstat.gui.mstatHierUI import Ui_Dialog as HierDialogGUI
    from mstat.dependencies.hier_clustering import getHier, plot_dendrogram
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class HierSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    progressComplete = QtCore.pyqtSignal(CompFlag, np.ndarray, np.ndarray, Exception)

class HierWorker(QtCore.QRunnable):
    def __init__(self, data):
        super(HierWorker, self).__init__()
        self.signals = HierSignals()
        self.data = data

    def run(self):
        time.sleep(1)

        features, labels = self.data
        norm_data = getTICNormalization(features)

        print("Hier Data", norm_data.shape)

        encoder = LabelEncoder().fit(labels)

        n = len(encoder.classes_)
        X = np.empty((n, features.shape[1]))

        for i, label in enumerate(encoder.classes_):
            class_data = features[(labels == label)]

            mean = np.mean(class_data, axis=0)
            X[i] = mean
        print(X.shape)

        # use relative distances of data to create clustering threshold
        tree = KDTree(X)
        nearest_dist, nearest_ind = tree.query(X, k=2)
        #cust_thresh = nearest_dist[0,1]

        print(f"Nearest sep: {nearest_dist[0,1]}")

        hdict, linkage_matrix = getHier(X, encoder)
        print(linkage_matrix)

        self.signals.progressComplete.emit(CompFlag.SUCCESS, linkage_matrix, encoder.classes_, Exception())

class HierCtrl(QtWidgets.QDialog):
    def __init__(self, main_ctrl, model_ctrl, main_gui):
        super().__init__()
        self.main_gui = main_gui
        self.model_ctrl = model_ctrl
        self.main_ctrl = main_ctrl

        self.dialog = HierDialogGUI()
        self.dialog.setupUi(self)
        self.dialog.buttonBox.accepted.connect(self.dialog_accepted)
        self.dialog.buttonBox.rejected.connect(self.dialog_rejected)

        training_keys = self.model_ctrl.training_dict.keys()

        self.show()

    def dialog_accepted(self):

        mzs, features, labels = self.model_ctrl.get_feature_data(DataRole.TRAINING)
        print('ALL DATA', features.shape)

        worker = HierWorker((features, labels))
        worker.setAutoDelete(True)
        worker.signals.progressChanged.connect(self.on_data_worker_update)
        worker.signals.progressComplete.connect(self.on_data_worker_complete)
        
        self.main_ctrl.threadpool.start(worker) 

    def on_data_worker_complete(self, comp_flag, linkage_matrix, classes, exc):
        print(f"\tEnding worker...")
        if comp_flag == CompFlag.SUCCESS:
            max_sep = linkage_matrix[-1, 2]
            norm_sep = linkage_matrix[:, 2] / max_sep
            linkage_matrix[:,2] = norm_sep

            plt.title("Hierarchical Clustering Dendrogram")
            plot_dendrogram(linkage_matrix, labels=classes, orientation='right', distance_sort='ascending')
            plt.tight_layout()
            #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.xlabel("Separation Distance (normalized to max separation)")
            #plt.yticks([])
            plt.grid()
            plt.show()
            
            print("Hierarchical Clustering is complete")
        else:
            self.main_gui.showError(f"Error with Hierarchical Clustering analysis \n{self} \n\n{exc}")
        
    def on_data_worker_update(self, result):
        #self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setValue(result)
        #self.worker_progress_list[row_num] = result
        #print(f"\t{result} {row_num}")
        #self.main_gui.statusprogress_bar.setValue(sum(self.worker_progress_list))
        pass

    def dialog_rejected(self):
        self.close()
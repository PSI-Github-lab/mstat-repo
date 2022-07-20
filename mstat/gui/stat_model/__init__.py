try:
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    import pandas as pd
    import numpy as np
    import time
    from sklearn.preprocessing import LabelEncoder
    from PyQt5 import QtCore
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.ScikitImports import StratifiedKFold, cross_validate
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class PCALDACtrl():
    from mstat.gui.stat_model.loading_funcs import add_data, clear_data, remove_data, move_data, on_data_worker_complete, on_data_worker_update
    from mstat.gui.stat_model.training_funcs import build_model, on_model_worker_complete, on_model_worker_update
    from mstat.gui.stat_model.testing_funcs import test_model, test_single_file, on_test_worker_complete, on_test_worker_update
    from mstat.gui.stat_model.learning_curves import plot_learning_curve

    def __init__(self, main_ctrl, main_gui, pca_dim=-1, bin_size=-1, low_lim=-1, up_lim=-1):
        self.main_gui = main_gui
        self.main_ctrl = main_ctrl
        self.training_dict = {}
        self.testing_dict = {}
        self.le_classes = None
        self.pca_dim = pca_dim
        self.bin_size = bin_size
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.rnd_state = 42

        self.progress_total = .0
        self.worker_progress_list = []
        self.num_processes = 0
        self.completed_processes = 0

        self.trained_flag = False

    def set_bin_size(self, bin_size):
        self.bin_size = bin_size

    def isTrained(self):
        return self.trained_flag

    def save_model(self, model_name):
        """
        save model to external file (adds .model extension to the path name)
        """
        if self.isTrained():
            try:
                joblib.dump(self.model, f"{model_name}.model")
            except Exception as exc:
                return CompFlag.FAILURE, exc
            return CompFlag.SUCCESS, Exception()
        return CompFlag.NODATA, Exception()

    def load_model(self, file_name):
        self.pred_est, self.outl_est, meta_info = joblib.load(file_name)
        self.model = (self.pred_est, self.outl_est, meta_info)

        self.pca_dim = meta_info['pca_dim']
        self.bin_size = meta_info['bin_size']
        self.low_lim = meta_info['low_lim']
        self.up_lim = meta_info['up_lim']
        self.meta_info = meta_info

        print("LOADED MODEL PARAMETERS")
        print('class names', meta_info['training_files'].keys())
        print('pca_dim', meta_info['pca_dim'])
        print('bin_size', meta_info['bin_size'])
        print('low_lim', meta_info['low_lim'])
        print('up_lim', meta_info['up_lim'])

        self.le_classes = list(meta_info['training_files'].keys())
        self.lda_dim = len(self.le_classes) - 1
        self.trained_flag = True

        self.main_gui.main_view.pcadim_spin.setValue(self.pca_dim)

    def reset_model(self):
        self.model = None
        self.trained_flag = False
        self.le_classes = None
        self.bin_size = -1
        self.low_lim = -1
        self.up_lim = -1

    def get_model(self):
        """
        return trained model in a tuple as (pred_est, outl_est, meta_info)
        """
        try:
            return self.model
        except Exception as e:
            return None

    def learning_curve(self):
        _, training_features, training_labels = extract_feature_data(self.training_dict, self.bin_size, self.low_lim, self.up_lim)
        cv = StratifiedKFold(n_splits=20)
        self.plot_learning_curve(self.pred_est, '', training_features, training_labels, self.pca_dim, axes=None, cv=cv, train_sizes=np.linspace(0.2, 1, 10))

    def get_loadings(self, pcalda_flag : bool, axis=0) -> tuple:
        if self.isTrained():
            pca = self.pred_est['dim']
            pca_loadings = pca.components_
            if pcalda_flag:
                lda = self.pred_est['lda']
                lda_loadings = lda.coef_
                loadings = np.dot(lda_loadings, pca_loadings)
            else:
                loadings = pca_loadings
            #print('LOADINGS', loadings.shape)
            #print('BINS', self.low_lim, self.up_lim, self.bin_size)
            binned_mzs, _ = calcBins(self.low_lim, self.up_lim, self.bin_size)
            return binned_mzs, loadings[axis, :]

    def getPCALDATrainingScores(self, pcalda_flag : bool, pcx=0, pcy=1, label_unknowns=True) -> tuple:
        #print(f"pcalda_dict: {self.training_dict.keys()}")
        if self.trained_flag:
            if pcalda_flag:
                scores, labels, feature_data, mzs, sample_names = self.getLDAScores(self.training_dict)
            else:
                scores, labels, feature_data, mzs, sample_names = self.getPCAScores(self.training_dict)
            if scores is None:
                return (None, None, None, None, None, None)
            scores_xy = np.empty((scores.shape[0], 2))
            if not label_unknowns:
                labels = ['unknown' if s not in self.le_classes else s for s in labels]
            scores_xy[:,0] = scores[:,pcx]
            scores_xy[:,1] = scores[:,pcy]
            return scores_xy, labels, self.le_classes, feature_data, mzs, sample_names
        else:
            self.main_gui.showError("PCA-LDA Model has not been trained yet.")
            return (None, None, None, None, None, None)

    def getPCALDATestingScores(self, pcalda_flag : bool, pcx=0, pcy=1, label_unknowns=True) -> tuple:
        if self.trained_flag:
            if self.testing_dict:
                if pcalda_flag:
                    scores, labels, feature_data, mzs, sample_names = self.getLDAScores(self.testing_dict)
                else:
                    scores, labels, feature_data, mzs, sample_names = self.getPCAScores(self.testing_dict)
                if scores is None:
                    return (None, None, None, None, None, None)
                scores_xy = np.empty((scores.shape[0], 2))
                if not label_unknowns:
                    labels = ['unknown' if s not in self.le_classes else s for s in labels]
                scores_xy[:,0] = scores[:,pcx]
                scores_xy[:,1] = scores[:,pcy]
                return scores_xy, labels, self.le_classes, feature_data, mzs, sample_names
            else:
                self.main_gui.showError("PCA-LDA Controller has no testing data.")
                return (None, None, None, None, None, None)
        else:
            self.main_gui.showError("PCA-LDA Model has not been trained yet.")
            return (None, None, None, None, None, None)

    def getPCAScores(self, data_dict):
        try:
            mzs, data, labels = extract_feature_data(data_dict, self.bin_size, self.low_lim, self.up_lim)
            meta = extract_meta_data(data_dict)
            norm_data = getTICNormalization(data)
            print(f"Norm data shape {norm_data.shape} {len(norm_data.shape)}")
            sample_names = [entry['filename'] for entry in meta]
            if len(norm_data.shape) == 1:
                return self.pred_est[:-1].transform(norm_data.reshape(1, -1)), labels, data, mzs, np.array(sample_names)
            return self.pred_est[:-1].transform(norm_data), labels, data, mzs, np.array(sample_names)
        except Exception as exc:
            print(f'From {os.path.basename(__file__)}')
            print(exc)
            self.main_gui.showError(f"Cannot process data to match current model parameters.\n\n{exc}")
            
            return None, None

    def getLDAScores(self, data_dict):
        try:
            mzs, data, labels = extract_feature_data(data_dict, self.bin_size, self.low_lim, self.up_lim)
            meta = extract_meta_data(data_dict)
            norm_data = getTICNormalization(data)
            print(f"Norm data shape {norm_data.shape} {len(norm_data.shape)}")
            sample_names = [entry['filename'] for entry in meta]
            if len(norm_data.shape) == 1:
                return self.pred_est.transform(norm_data.reshape(1, -1)), labels, data, mzs, np.array(sample_names)
            return self.pred_est.transform(norm_data), labels, data, mzs, np.array(sample_names)
        except Exception as exc:
            print(f'From {os.path.basename(__file__)}')
            print(exc)
            self.main_gui.showError(f"Cannot process data to match current model parameters.\n\n{exc}")
            return None, None

    def get_feature_data(self, role : DataRole):
        if role == DataRole.TRAINING:
            return extract_feature_data(self.training_dict, self.bin_size, self.low_lim, self.up_lim)
        elif role == DataRole.TESTING:
            return extract_feature_data( self.testing_dict, self.bin_size, self.low_lim, self.up_lim)

    def get_meta_data(self, role : DataRole):
        if role == DataRole.TRAINING:
            return extract_meta_data(self.training_dict)
        elif role == DataRole.TESTING:
            return extract_meta_data(self.testing_dict)

    def get_num_pca_dim(self):
        return self.pca_dim

    def get_num_lda_dim(self):
        return self.lda_dim

    def get_num_classes(self):
        if self.le_classes:
            return len(self.le_classes)
        return 0
        



try:
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import TruncatedSVD, PCA, FactorAnalysis
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

class ModelWorkerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal()
    progressComplete = QtCore.pyqtSignal(tuple, LabelEncoder, list, int, float, float, float)

class ModelWorker(QtCore.QRunnable):
    def __init__(self, training_dict, pca_dim, bin_size, low_lim, up_lim, do_diff, diff_order, rnd_state=42):
        super(ModelWorker, self).__init__()
        self.signals = ModelWorkerSignals()
        self.pca_dim, self.bin_size, self.low_lim, self.up_lim, self.do_diff, self.diff_order = pca_dim, bin_size, low_lim, up_lim, do_diff, diff_order
        self.training_dict = training_dict
        self.rnd_state = rnd_state

    def run(self):
        self.create_model()
        norm_data, labels, trans_exc = self.transform_data()
        model_tuple, train_excs = self.train_model(norm_data, labels)
        encoder = LabelEncoder().fit(labels)
        self.signals.progressComplete.emit(model_tuple, encoder, [train_excs[0], train_excs[1], trans_exc], self.pca_dim, self.bin_size, self.low_lim, self.up_lim)

    def create_model(self):
        # CREATE MODEL HERE
        steps = [
            #('dim', FactorAnalysis(n_components=self.pca_dim, random_state=self.rnd_state, rotation='varimax')),
            ('dim', PCA(n_components=self.pca_dim, random_state=self.rnd_state)),
            ('lda', LDA(store_covariance=True)),
            ]
        self.pred_est = Pipeline(steps)

        # create outlier estimator and train it, for now it is unused
        self.outl_est = Pipeline(
        [
        ('lof', LocalOutlierFactor(novelty=True, n_neighbors=20)),
        ])

    def transform_data(self):
        # load the data for training, lower and upper m/z limits are determined by the training data
        #try:
        meta = extract_meta_data(self.training_dict)
        # determine model m/z limits
        self.low_lim = max(float(meta[0]['lowlim']), self.low_lim)
        self.up_lim = min(float(meta[0]['uplim']), self.up_lim)
        mzs, data, labels = extract_feature_data(self.training_dict, self.bin_size, self.low_lim, self.up_lim)
        #except Exception as exc:
        #    print(f'From {os.path.basename(__file__)}')
        #    self.main_gui.showError(f"transform Cannot process data.\n\n{exc}")
        #    return None, None, exc
        if self.do_diff:
            for _ in range(self.diff_order):
                data = np.gradient(data, axis=1)
        norm_data = getTICNormalization(data)
        return norm_data, labels, Exception()

    def train_model(self, data, labels):
        # fit the model with training data
        train_exc = None
        outli_exc = None
        try:
            self.pred_est.fit(data, labels)
        except Exception as exc1:
            train_exc = exc1
        try:
            self.outl_est.fit(data, labels)
        except Exception as exc2:
            outli_exc = exc2

        # meta data from model parameters and info about training file and random seed
        model_dict = self.pred_est.get_params()
        model_dict['training_files'] = self.training_dict
        model_dict['pca_dim'] = self.pca_dim
        model_dict['bin_size'] = self.bin_size
        model_dict['low_lim'] = self.low_lim
        model_dict['up_lim'] = self.up_lim
        model_dict['random_seed'] = self.rnd_state

        # cross-validation
        try:
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.rnd_state)
            cv_results = cross_validate(self.pred_est, data, labels, cv=cv, scoring='balanced_accuracy')
            model_dict['cv_results'] = cv_results
        except Exception as exc:
            model_dict['cv_results'] = CompFlag.FAILURE

        meta_info = model_dict

        return (self.pred_est, self.outl_est, meta_info), [train_exc, outli_exc]

def on_model_worker_update(self):
    pass

def on_model_worker_complete(self, model, model_encoder, exceptions, pca_dim, bin_size, low_lim, up_lim):
    print(f"\tEnding worker...")
    self.num_processes -= 1
    self.pca_dim, self.bin_size, self.low_lim, self.up_lim = pca_dim, bin_size, low_lim, up_lim
    self.model = model
    self.pred_est, self.outl_est, _ = model
    self.model_encoder = model_encoder
    self.le_classes = list(self.model_encoder.classes_)
    self.lda_dim = len(self.le_classes) - 1

    print('TRAINING COMPLETE', self.le_classes, self.lda_dim)

    if exceptions[0] is None and self.num_processes == 0:
        self.trained_flag = True
        self.progress_total = 0
        self.main_ctrl.change_model_option()
        self.main_ctrl.gui.showInfo("Model training is complete!")
    else:
        self.main_ctrl.gui.showInfo(f"Model training is completed with the following exceptions:\n\n{exceptions}")
        self.main_ctrl.set_state(ControlStates.READY)

def build_model(self, pca_dim, do_diff, diff_order):
    self.pca_dim = pca_dim
    if self.bin_size < 0:
        self.bin_size = 1.0
    if self.low_lim < 0:
        self.low_lim = -np.inf
    if self.up_lim < 0:
        self.up_lim = np.inf
    self.do_diff, self.diff_order = do_diff, diff_order

    print('MODEL PARAMETERS', self.pca_dim, self.bin_size, self.low_lim, self.up_lim)
    print('MODEL PARAMETERS (CONT)', self.do_diff, self.diff_order)

    worker = ModelWorker(self.training_dict, self.pca_dim, self.bin_size, self.low_lim, self.up_lim, self.do_diff, self.diff_order)
    worker.setAutoDelete(True)
    worker.signals.progressChanged.connect(self.on_model_worker_update)
    worker.signals.progressComplete.connect(self.on_model_worker_complete)
    self.main_ctrl.threadpool.start(worker)
    #self.worker_progress_list.append(0)
    self.progress_total += 1
    self.num_processes += 1
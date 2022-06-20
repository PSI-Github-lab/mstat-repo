try:
    
    #from dependencies.ScikitImports import *
    import joblib
    import bisect
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    import pandas as pd
    import numpy as np
    import time
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import binned_statistic
    from PyQt5 import QtCore
    from mstat.dependencies.file_conversion.MZMLDirectory import MZMLDirectory
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.ScikitImports import StratifiedKFold, cross_validate
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class DataWorkerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int, int)
    progressComplete = QtCore.pyqtSignal(int, tuple, str, str, str, Exception)

class DataWorker(QtCore.QRunnable):
    def __init__(self, class_name, dir, role):
        super(DataWorker, self).__init__()
        self.signals = DataWorkerSignals()
        self.class_name = class_name
        self.dir = dir
        self.role = role

    def run(self):
        with open(npy_file_name(self.dir), 'rb') as f:
            intens = np.load(f, allow_pickle=True)
            mzs = np.load(f, allow_pickle=True)
            meta = np.load(f, allow_pickle=True)
            
        data = intens, mzs, meta

        if data[0].size == 0:
            self.signals.progressComplete.emit(0, data, self.class_name, self.dir, self.role, Exception())
        else:
            #label = [self.class_name] * len(df.values)
            #df.insert(0, 'label', label, True)
            self.signals.progressComplete.emit(1, data, self.class_name, self.dir, self.role, Exception())
                
        #except Exception:
        #    self.signals.progressComplete.emit(2, None, self.class_name, self.dir, self.role)

class PCALDACtrl():
    def __init__(self, main_ctrl, main_gui, pca_dim=-1, bin_size=-1, low_lim=-1, up_lim=-1):
        self.main_gui = main_gui
        self.main_ctrl = main_ctrl
        self.training_dict = {}
        self.testing_dict = {}
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

    def build_model(self, pca_dim) -> None:
        self.pca_dim = pca_dim
        if self.bin_size < 0:
            self.bin_size = 1.0
        if self.low_lim < 0:
            self.low_lim = -np.inf
        if self.up_lim < 0:
            self.up_lim = np.inf

        print('PARAMETERS', self.pca_dim, self.bin_size, self.low_lim, self.up_lim)

        # CREATE MODEL HERE
        steps = [
            ('dim', PCA(n_components=pca_dim, random_state=self.rnd_state)),
            ('lda', LDA(store_covariance=True)),
            ]
        self.pred_est = Pipeline(steps)

        # create outlier estimator and train it, for now it is unused
        self.outl_est = Pipeline(
        [
        ('lof', LocalOutlierFactor(novelty=True, n_neighbors=20)),
        ])

        # load the data for training, lower and upper m/z limits are determined by the training data
        #try:
        meta = self.get_meta_data(self.training_dict)
        # determine model m/z limits
        self.low_lim = max(float(meta[0]['lowlim']), self.low_lim)
        self.up_lim = min(float(meta[0]['uplim']), self.up_lim)
        mzs, data, labels = self.get_feature_data(self.training_dict, self.bin_size, self.low_lim, self.up_lim)
        #except Exception as exc:
        #    self.main_gui.showError(f"Cannot process data.\n\n{exc}")
        print(data.shape)
        print("DATA PROCESSED CORRECTLY")
        # normalize data and train the model
        norm_data = self.getTICNormalization(data)
        return self.train_model(norm_data, labels)

    def train_model(self, data, labels):
        # fit the model with training data
        train_exc = None
        outli_exc = None
        try:
            self.pred_est.fit(data, labels)
        except Exception as exc1:
            train_exc = exc1
        try:
            pass#self.outl_est.fit(data, labels)
        except Exception as exc2:
            outli_exc = exc2
        
        self.my_encoder = LabelEncoder().fit(labels)
        le_classes = list(self.my_encoder.classes_)
        self.lda_dim = len(le_classes) - 1
        le_classes.append('unknown')
        self.my_encoder.classes_ = np.array(le_classes)

        # meta data from model parameters and info about training file and random seed
        model_dict = self.pred_est.get_params()
        model_dict['training_files'] = self.training_dict
        model_dict['bin_size'] = self.bin_size
        model_dict['random_seed'] = self.rnd_state
        meta_info = model_dict

        # cross-validation
        try:
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.rnd_state)
            cv_results = cross_validate(self.pred_est, data, labels, cv=cv, scoring='balanced_accuracy')
            model_dict['cv_results'] = cv_results
        except Exception as exc:
            model_dict['cv_results'] = CompFlag.FAILURE

        self.model = (self.pred_est, self.outl_est, meta_info)
        if train_exc is None:
            self.trained_flag = True
        return train_exc, outli_exc

    def bin_data(self, mzs, intens, meta, bin_size, low, up):
        #print('BINNING', mzs.shape, intens.shape)
        bins, num_bins = calcBins(low, up, bin_size)
        #print('BINNING lims', low, up)
        stats, bin_edges, _ = binned_statistic(mzs, intens, 'mean', bins=num_bins, range=(low, up))
        stats[np.isnan(stats)] = 0
        #print(stats[:5], sum(row), sum(stats))
        #new_row = np.concatenate(([sum(stats)], stats)) 
        #rows.append(stats)
        return bin_edges + bin_size/2, np.array([stats])

    def on_progressComplete(self, result, data, class_name, path, role):
        print(f"\tEnding worker...")
        if result == 1:
            self.num_processes -= 1
            intens, mzs, meta = data
            if role == "training":
                try:
                    self.training_dict[class_name].update({path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}})
                except KeyError:
                    self.training_dict[class_name] = {path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}}
            elif role == "testing":
                try:
                    self.testing_dict[class_name].update({path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}})
                except KeyError:
                    self.testing_dict[class_name] = {path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}}

            if self.isTrained() and (float(meta[0]['lowlim']) != self.low_lim or float(meta[0]['uplim']) != self.up_lim):
                self.main_gui.showInfo("m/z limits of these data do not match the limits in the training data. Data will be padded with zero values to try to match training data m/s limits.\n\nDelete the model to stop seeing this message.")
            #print("pcalda key", self.training_dict.keys())
            #for key in self.training_dict:
            #    for j in self.training_dict[key]:
            #        print(self.training_dict[key][j][0,:10])
               
            #print("pcalda model dict", self.training_dict)
            
            if self.num_processes == 0:
                print(f"--- Ran in {time.time() - self.start_time} seconds ---") 
                print("All processes completed")
                self.progress_total = 0
                #self.training_frame = pd.concat(self.training_frames)
                #print(self.training_dict)

                # remove any rows with all zero values
                #self.training_frame.drop(self.training_frame[self.training_frame['total'] == 0].index, inplace=True)

                # extract feature data and labels
                #feature_data = getFeatureData(self.training_frame)
                #class_labels = getLabelData(self.training_frame)

                # create standard bin labels just in case
                #self.bins = [float(bin) for bin in feature_cols]

                #self.trainModel(feature_data, class_labels)
                #print(self.pred_est)
                #output = self.main_gui.showInfo("Model training is complete!")
                self.main_ctrl.set_state(ControlStates.PLOTTING)
        elif result == 0:
            self.main_gui.showError(f"No data converted from \n{self}")
        else:
            self.main_gui.showError(f"Unknown error occured for \n{self}")

    def on_progressUpdate(self, result, result2):
        #self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setValue(result)
        #self.worker_progress_list[row_num] = result
        #print(f"\t{result} {row_num}")
        #self.main_gui.statusprogress_bar.setValue(sum(self.worker_progress_list))
        pass

    def save_model(self, model_name):
        """
        save model to external file (adds .model extension to the path name)
        """
        try:
            joblib.dump(self.model, f"{model_name}.model")
        except Exception as exc:
            return CompFlag.FAILURE, exc
        return CompFlag.SUCCESS, Exception()
    
    def reset_model(self):
        self.model = None
        self.trained_flag = False
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

    def addData(self, class_name : str, new_data_path : str, role : str) -> None:
        worker = DataWorker(class_name, new_data_path, role)
        worker.setAutoDelete(True)
        worker.signals.progressChanged.connect(self.on_progressUpdate)
        worker.signals.progressComplete.connect(self.on_progressComplete)
        #print(f"\tMaking frame for {new_mzml_path}...")
        self.main_ctrl.threadpool.start(worker)
        self.worker_progress_list.append(0)
        self.progress_total += 1
        self.num_processes += 1

    def clearTrainingData(self):
        self.training_dict = {}

    def addTrainingData(self, class_name : str, new_data_path : str) -> None:
        # create a worker to convert the data in the folder into a numpy array
        self.start_time = time.time()
        self.addData(class_name, new_data_path, "training")

    def removeTrainingData(self, class_name : str, path : str) -> None:
        #print(self.training_dict.keys())
        self.training_dict[class_name].pop(path)
        if self.training_dict[class_name] == {}:
            self.training_dict.pop(class_name)

    def moveTrainingData(self, old_class : str, new_class : str, path : str) -> int:
        try:
            temp = self.training_dict[old_class].pop(path)
        except KeyError:
            return CompFlag.FAILURE
        try:
            self.training_dict[new_class].update({path : temp})
        except KeyError:
            self.training_dict[new_class] = {path : temp}
        if self.training_dict[old_class] == {}:
            self.training_dict.pop(old_class)  
        return CompFlag.SUCCESS

    def clearTestingData(self):
        self.testing_dict = {}

    def addTestingData(self, class_name, new_data_path):
        # create a worker to convert the data in the folder into a numpy array
        self.start_time = time.time()
        self.addData(class_name, new_data_path, "testing")

    def removeTestingData(self, class_name, path):
        self.testing_dict[class_name].pop(path)
        if self.testing_dict[class_name] == {}:
            self.testing_dict.pop(class_name)

    def moveTestingData(self, old_class, new_class, path):
        try:
            temp = self.testing_dict[old_class].pop(path)
        except KeyError:
            return CompFlag.FAILURE
        try:
            self.testing_dict[new_class].update({path : temp})
        except KeyError:
            self.testing_dict[new_class] = {path : temp}
        if self.testing_dict[old_class] == {}:
            self.testing_dict.pop(old_class)  
        return CompFlag.SUCCESS

    def get_training_meta_data(self):
        return self.get_meta_data(self.training_dict)

    def get_testing_meta_data(self):
        return self.get_meta_data(self.testing_dict)

    def isTrained(self):
        return self.trained_flag

    def getPCALDATrainingScores(self, pcalda_flag : bool, pcx=0, pcy=1) -> tuple:
        print(f"pcalda_dict: {self.training_dict.keys()}")
        if self.trained_flag:
            if pcalda_flag:
                data, labels = self.getLDAScores(self.training_dict)
            else:
                data, labels = self.getPCAScores(self.training_dict)
            if data is None:
                return (None, None, None)
            combined_data = np.empty((data.shape[0], 3))
            labels = ['unknown' if s not in self.my_encoder.classes_ else s for s in labels]
            combined_data[:,0] = data[:,pcx]
            combined_data[:,1] = data[:,pcy]
            combined_data[:,2] = self.my_encoder.transform(labels)
            u_labels = np.unique(labels)
            u_labels_pairs = {(label, int(self.my_encoder.transform([label]))) for label in u_labels}
            return combined_data, labels, u_labels_pairs
        else:
            self.main_gui.showError("PCA-LDA Model has not been trained yet.")

    def getPCALDATestingScores(self, pcalda_flag : bool, pcx=0, pcy=1) -> tuple:
        if self.trained_flag:
            if self.testing_dict:
                if pcalda_flag:
                    data, labels = self.getLDAScores(self.testing_dict)
                else:
                    data, labels = self.getPCAScores(self.testing_dict)
                if data is None:
                    return (None, None, None)
                combined_data = np.empty((data.shape[0], 3))
                labels = ['unknown' if s not in self.my_encoder.classes_ else s for s in labels]
                combined_data[:,0] = data[:,pcx]
                combined_data[:,1] = data[:,pcy]
                combined_data[:,2] = self.my_encoder.transform(labels)
                u_labels = np.unique(labels)
                u_labels_pairs = {(label, int(self.my_encoder.transform([label]))) for label in u_labels}
                return combined_data, labels, u_labels_pairs 
            else:
                self.main_gui.showError("PCA-LDA Controller has no testing data.")
                return (None, None, None)
        else:
            self.main_gui.showError("PCA-LDA Model has not been trained yet.")
            return (None, None, None)

    def getPCAScores(self, data_dict):
        #feature_data = getFeatureData(frame)
        try:
            meta = self.get_meta_data(data_dict)
            self.low_lim = max(float(meta[0]['lowlim']), self.low_lim)
            self.up_lim = min(float(meta[0]['uplim']), self.up_lim)
            mzs, data, labels = self.get_feature_data(self.training_dict, self.bin_size, self.low_lim, self.up_lim)
            
            #_, binned_data = self.bin_data(mzs, data, meta, self.bin_size, self.low_lim, self.up_lim)
            norm_data = self.getTICNormalization(data)
            print(f"PCA data shape {norm_data.shape}")
            
            return self.pred_est[:-1].transform(norm_data), labels
        except Exception as exc:
            self.main_gui.showError(f"Cannot process data to match current model parameters.\n\n{exc}")
            return None, None

    def getLDAScores(self, data_dict):
        try:
            mzs, data, labels = self.get_feature_data(data_dict, self.bin_size, self.low_lim, self.up_lim)
            meta = self.get_meta_data(data_dict)
            #edges, binned_data = self.bin_data(mzs, data, meta, self.bin_size, self.low_lim, self.up_lim)
            norm_data = self.getTICNormalization(data)
            print(f"PCA-LDA data shape {norm_data.shape}")
            #print(edges[0], edges[-1])
            
            return self.pred_est.transform(norm_data), labels
        except Exception as exc:
            self.main_gui.showError(f"Cannot process data to match current model parameters.\n\n{exc}")
            return None, None

    def get_feature_data(self, data_dict : dict, bin_size : float, low_lim : float, up_lim : float):
        """
        Get all of the feature data in a give data dictionary
        """
        intens_dict = {}
        labels = []
        for key in data_dict:
            intens_list = [data_dict[key][x]['intens'] for x in data_dict[key]]
            mzs_list = [data_dict[key][x]['mzs'] for x in data_dict[key]]
            binned_list = []
            #print(len(mzs_list), len(intens_list))
            for mzs_matrix, intens_matrix in zip(mzs_list, intens_list):
                for mzs, intens in zip(mzs_matrix, intens_matrix): 
                    #print(mzs.shape, intens.shape)
                    bin_edges, binned_data = self.bin_data(mzs, intens, None, bin_size, low_lim, up_lim)
                    binned_list.append(binned_data)
            intens_dict[key] = np.concatenate(binned_list, 0)
            
        for key in intens_dict:
            #print(f"this should be a class: {key}")
            labels += [key] * intens_dict[key].shape[0]
        labels = np.array(labels)
        #print(temp_dict, len(labels))
        #mzs = np.concatenate([mzs_dict[x] for x in mzs_dict], 0)
        intens = np.concatenate([intens_dict[x] for x in intens_dict], 0)
        print('BIN DATA SHAPE', bin_edges[:-1].shape, intens.shape, labels.shape)
        return bin_edges[:-1], intens, labels
        
    def get_meta_data(self, data_dict : dict):
        """
        Get numpy array of ordered metadata in a given data dictionary
        """
        meta_dict = {}
        for key in data_dict:
            meta_dict[key] = np.concatenate([data_dict[key][x]['metadata'] for x in data_dict[key]], 0)
        #print('META DICT', meta_dict)
        return np.concatenate([meta_dict[x] for x in meta_dict], 0)

    def get_num_pca_dim(self):
        return self.pca_dim

    def get_num_lda_dim(self):
        return self.lda_dim

    def getTICNormalization(self, data):
        rows, cols = data.shape
        norm_data = np.empty(0)

        for row in range(rows):
            temp = np.empty(0)
            total = sum(data[row])
            #for element in data[row,1:]:
            temp = data[row].astype('float') / total
            
            try:
                norm_data = np.vstack((norm_data, temp))
            except ValueError:
                norm_data = temp

        return norm_data
        



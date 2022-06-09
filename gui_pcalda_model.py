try:
    from helper_funcs import *
    #from dependencies.ScikitImports import *
    import joblib
    import bisect
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from dependencies.file_conversion.MZMLDirectory import MZMLDirectory
    from model_TestModel import *
    import pandas as pd
    import numpy as np
    import time
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import binned_statistic
    from PyQt5 import QtCore
except ModuleNotFoundError as e:
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
        #try:
            #array, header = self.dir.createArray(self.bin_size, self.low_lim, self.up_lim)
        with open(rf'{self.dir}\{os.path.basename(self.dir)}.npy', 'rb') as f:
            intens = np.load(f, allow_pickle=True)
            mzs = np.load(f, allow_pickle=True)
            meta = np.load(f, allow_pickle=True)
            
        #binned_intens, bins, meta 
        #data = np.array(rows), bins + self.bin_size/2, meta
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
    def __init__(self, main_ctrl, main_gui, pca_dim=-1, bin_size=-1):
        self.main_gui = main_gui
        self.main_ctrl = main_ctrl
        self.training_dict = {}
        self.testing_dict = {}
        self.pca_dim = pca_dim
        self.rnd_state = 42

        self.progress_total = .0
        self.worker_progress_list = []
        self.num_processes = 0
        self.completed_processes = 0

        self.bin_size = bin_size
        self.low_lim = -np.inf
        self.up_lim = np.inf
        self.trained_flag = False

    def set_bin_size(self, bin_size):
        self.bin_size = bin_size

    def build_model(self, pca_dim) -> None:
        self.pca_dim = pca_dim
        #print(f"PCA dimensions: {pca_dim}")
        #print(f"Training Dict: {self.training_dict}")
        #print(f"Testing Dict: {self.testing_data}")

        # CREATE MODEL HERE
        steps = [
            ('dim', PCA(n_components=pca_dim, random_state=self.rnd_state)),
            ('lda', LDA(store_covariance=True)),
            ]
        self.pred_est = Pipeline(steps)

        # create outlier estimator and train
        self.outl_est = Pipeline(
        [
        ('lof', LocalOutlierFactor(novelty=True, n_neighbors=20)),
        ])
        #outl_est = None

        # combine the data
        #print(self.training_dict.keys())  
        #for i in self.training_dict:
        #    for j in self.training_dict[i]:
        #        print(self.training_dict[i][j].keys())
        
        mzs, data, labels = self.get_feature_data(self.training_dict)
        meta = self.get_meta_data(self.training_dict)
        self.low_lim = float(meta[0]['lowlim'])
        self.up_lim = float(meta[0]['uplim'])
        self.model_bins, binned_data = self.bin_data(mzs, data, meta, bin_size=self.bin_size)

        # normalize data and train the model
        norm_data = self.getTICNormalization(binned_data)
        self.train_model(norm_data, labels)

    def train_model(self, data, labels):
        # fit the model with training data
        self.pred_est.fit(data, labels)
        self.outl_est.fit(data, labels)
        self.my_encoder = LabelEncoder().fit(labels)
        le_classes = list(self.my_encoder.classes_)
        self.lda_dim = len(le_classes) - 1
        le_classes.append('unknown')
        self.my_encoder.classes_ = np.array(le_classes)

        # meta data from model parameters and info about training file and random seed
        model_dict = self.pred_est.get_params()
        model_dict['training_files'] = self.training_dict
        model_dict['bin_size'] = self.model_bins[1] - self.model_bins[0]
        model_dict['random_seed'] = self.rnd_state
        meta_info = model_dict

        # cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.rnd_state)
        cv_results = cross_validate(self.pred_est, data, labels, cv=cv, scoring='balanced_accuracy')
        model_dict['cv_results'] = cv_results

        self.model = (self.pred_est, self.outl_est, meta_info)
        self.trained_flag = True

    def bin_data(self, mzs, intens, meta, bin_size=1.0):
        #print('BINNING', mzs.shape, intens.shape)
        low = self.low_lim#float(meta[0]['lowlim'])
        up = self.up_lim#float(meta[0]['uplim'])
        #print(low, up, bin_size)
        bins, num_bins = calcBins(low, up, bin_size)
        rows = []

        for row in intens:
            stats, bin_edges, _ = binned_statistic(mzs, row, 'mean', bins=num_bins, range=(low, up))
            stats[np.isnan(stats)] = 0
            #print(stats[:5], sum(row), sum(stats))
            new_row = np.concatenate(([sum(stats)], stats)) 
            rows.append(stats)

        return bin_edges + bin_size/2, np.array(rows)

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
                self.main_gui.showInfo("m/z limits of these data do not match the limits in the training data. Data has been padded with zero values to match training data m/s limits.")
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

    def saveModel(self, model_name):
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

        joblib.dump((pred_est, outl_est, meta_info), name + ".model")
        print(f"Generated and saved new model as {name}.model\n")

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
            return 1
        try:
            self.training_dict[new_class].update({path : temp})
        except KeyError:
            self.training_dict[new_class] = {path : temp}
        if self.training_dict[old_class] == {}:
            self.training_dict.pop(old_class)  
        return 0

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
            return 1
        try:
            self.testing_dict[new_class].update({path : temp})
        except KeyError:
            self.testing_dict[new_class] = {path : temp}
        if self.testing_dict[old_class] == {}:
            self.testing_dict.pop(old_class)  
        return 0

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
        mzs, data, labels = self.get_feature_data(data_dict)
        meta = self.get_meta_data(data_dict)
        _, binned_data = self.bin_data(mzs, data, meta, bin_size=self.bin_size)
        norm_data = self.getTICNormalization(binned_data)
        print(f"PCA data shape {norm_data.shape}")
        
        return self.pred_est[:-1].transform(norm_data), labels

    def getLDAScores(self, data_dict):
        mzs, data, labels = self.get_feature_data(data_dict)
        meta = self.get_meta_data(data_dict)
        edges, binned_data = self.bin_data(mzs, data, meta, bin_size=self.bin_size)
        norm_data = self.getTICNormalization(binned_data)
        print(f"PCA-LDA data shape {norm_data.shape}")
        print(edges[0], edges[-1])
        
        return self.pred_est.transform(norm_data), labels

    def get_feature_data(self, data_dict : dict):
        """
        Get all of the feature data in a give data dictionary
        """
        #print('DATA DICT', data_dict)
        intens_dict = {}
        labels = []
        for key in data_dict:
            intens_dict[key] = np.concatenate([data_dict[key][x]['intens'] for x in data_dict[key]], 0)
            mzs = data_dict[key][list(data_dict[key].keys())[0]]['mzs']
        for key in intens_dict:
            #print(f"this should be a class: {key}")
            labels += [key] * intens_dict[key].shape[0]
        #print(temp_dict, len(labels))
        #mzs = np.concatenate([mzs_dict[x] for x in mzs_dict], 0)
        #print('MZS SHAPE', mzs.shape)
        intens = np.concatenate([intens_dict[x] for x in intens_dict], 0)
        return mzs, intens, labels

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
        



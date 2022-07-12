try:
    import numpy as np
    import time
    from PyQt5 import QtCore
    from mstat.dependencies.helper_funcs import *
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class DataWorkerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int, int)
    progressComplete = QtCore.pyqtSignal(CompFlag, tuple, str, str, DataRole, Exception)

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
            self.signals.progressComplete.emit(CompFlag.FAILURE, data, self.class_name, self.dir, self.role, Exception())
        else:
            self.signals.progressComplete.emit(CompFlag.SUCCESS, data, self.class_name, self.dir, self.role, Exception())

def on_data_worker_complete(self, result, data, class_name, path, role):
    print(f"\tEnding worker...")
    if result == CompFlag.SUCCESS:
        self.num_processes -= 1
        intens, mzs, meta = data
        if role == DataRole.TRAINING:
            try:
                self.training_dict[class_name].update({path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}})
            except KeyError:
                self.training_dict[class_name] = {path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}}
        elif role == DataRole.TESTING:
            try:
                self.testing_dict[class_name].update({path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}})
            except KeyError:
                self.testing_dict[class_name] = {path : {'mzs' : mzs, 'intens' : intens, 'metadata' : meta}}

        if self.isTrained() and (float(meta[0]['lowlim']) != self.low_lim or float(meta[0]['uplim']) != self.up_lim):
            self.main_gui.showInfo("m/z limits of these data do not match the limits in the training data. Data will be padded with zero values to try to match training data m/s limits.\n\nDelete the model to stop seeing this message.")
        
        if self.num_processes <= 0:
            print(f"--- Ran in {time.time() - self.start_time} seconds ---") 
            print("All processes completed")
            self.progress_total = 0
            self.num_processes = 0
            self.main_ctrl.set_state(ControlStates.PLOTTING)
    elif result == CompFlag.FAILURE:
        self.main_gui.showError(f"No data converted from \n{self}")
    else:
        self.main_gui.showError(f"Unknown error occured for \n{self}")

def on_data_worker_update(self, result, result2):
    #self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setValue(result)
    #self.worker_progress_list[row_num] = result
    #print(f"\t{result} {row_num}")
    #self.main_gui.statusprogress_bar.setValue(sum(self.worker_progress_list))
    pass

def clear_data(self, role : DataRole):
    if role == DataRole.TRAINING:
        self.training_dict = {}
    elif role == DataRole.TESTING:
        self.testing_dict = {}

def add_data(self, class_name : str, new_data_path : str, role : DataRole) -> None:
    #print("add_data", class_name, new_data_path)
    self.start_time = time.time()
    worker = DataWorker(class_name, new_data_path, role)
    worker.setAutoDelete(True)
    worker.signals.progressChanged.connect(self.on_data_worker_update)
    worker.signals.progressComplete.connect(self.on_data_worker_complete)
    #print(f"\tMaking frame for {new_mzml_path}...")
    self.main_ctrl.threadpool.start(worker)
    self.worker_progress_list.append(0)
    self.progress_total += 1
    self.num_processes += 1

def remove_data(self, class_name : str, path : str, role : DataRole) -> None:
    if role == DataRole.TRAINING:
        self.training_dict[class_name].pop(path)
        if self.training_dict[class_name] == {}:
            self.training_dict.pop(class_name)
    elif role == DataRole.TESTING:
        self.testing_dict[class_name].pop(path)
        if self.testing_dict[class_name] == {}:
            self.testing_dict.pop(class_name)

def move_data(self, old_class : str, new_class : str, path : str, role : DataRole) -> CompFlag:
    if role == DataRole.TRAINING:
        data_dict = self.training_dict
    elif role == DataRole.TESTING:
        data_dict = self.testing_dict
    
    try:
        temp = data_dict[old_class].pop(path)
    except KeyError:
        return CompFlag.FAILURE
    try:
        data_dict[new_class].update({path : temp})
    except KeyError:
        data_dict[new_class] = {path : temp}
    if data_dict[old_class] == {}:
        data_dict.pop(old_class)  
    return CompFlag.SUCCESS
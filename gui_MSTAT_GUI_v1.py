import contextlib
import time


try:
    import sys
    import os
    from functools import partial
    import glob
    from helper_funcs import *
    from pathlib import Path
    from PyQt5 import QtCore, QtWidgets
    from gui_GUI import MainGUI
    from gui_Conversion import ConversionCtrl
    from gui_pcalda_model import PCALDACtrl
    from gui_plotter import DataPlot
    from gui_fileselector import FileTreeSelectorModel
    from gui_tableModel import TableModel
except ModuleNotFoundError as e:
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class MSTAT_Ctrl():
    def __init__(self, app, root_path=None, config_hdlr=None) -> None:
        self.app = app
        self.root_path = root_path
        self.config_hdlr = config_hdlr
        self.training_paths = []
        self.testing_paths = []
        self.training_dict = {}
        self.testing_dict = {}
        self.conversion_list = []
        self.ctrl_state = ControlStates.READY

        # Directory Selector model objects
        training_dir = self.config_hdlr.get_option('train directory')
        self.trainingdir_model = FileTreeSelectorModel(rootpath=training_dir)
        testing_dir = self.config_hdlr.get_option('test directory')
        self.testingdir_model = FileTreeSelectorModel(rootpath=testing_dir)
        
        # Model data table model object
        self.model_data_src = []
        self.model_data_headers = ['Folder', 'Class Name', '# Samples','Role']
        self.modeldata_model = TableModel(self, self.model_data_headers, self.model_data_src)
        self.modeldata_model.define_col_flags(1, 1)

        # Create the GUI
        self.gui = MainGUI(self, root_path)

        if self.config_hdlr.get_option('msfr installed') == 'False':
            if not self.gui.show_YN_dialog("Have you installed MSFileReader library from Thermo Scientific?", "MStat 64-bit Requirement"):
                self.gui.showInfo("MSFileReader has not been installed. Please install MSFileReader by following <a href='https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.0/faq/data-import-thermo-raw-need-libraries.aspx#downloading-libraries'>this guide</a>")
                quit()
            else:
                self.config_hdlr.set_option('msfr installed', 'True')
        
        self.gui.setupStatusBar()       # initialize the status bar at the bottom of the window
        self.gui.attachTTDirModels(self.trainingdir_model, self.testingdir_model, training_dir, testing_dir)   # include the directory models for traingin and testing data
        self.gui.setupModelDataTView(self.modeldata_model, self.model_data_headers)      # set of the selected model data table view
        self.gui.show()

        # create PCA-LDA model
        self.pcalda_ctrl = PCALDACtrl(self, self.gui, bin_size=float(self.gui.main_view.binsize_edit.text()))

        # set up the plot view
        self.plotter = DataPlot(self.gui)
        self.plotter.setup_plot()
        
        # Connect signals and slots
        QtCore.QMetaObject.connectSlotsByName(self.gui)
        self.gui.main_view.cleartrainingdata_button.clicked.connect(partial(self.clear_dir_checks, self.trainingdir_model))
        self.gui.main_view.cleartestingdata_button.clicked.connect(partial(self.clear_dir_checks, self.testingdir_model))
        self.gui.main_view.binsize_edit.textChanged.connect(self.change_bin_size)
        self.gui.main_view.trainmodel_button.clicked.connect(self.train_model) #partial(self.set_state, ControlStates.BUILDING))
        self.gui.main_view.modelinfo_button.clicked.connect(self.show_model_info)
        self.gui.statusbar_button.clicked.connect(self.convert_RAW)
        self.gui.main_view.extplot_button.clicked.connect(self.extern_plot)
        self.connect_plot_options()
        self.gui.main_view.actionAbout.triggered.connect(self.show_about_message)
        self.gui.main_view.actionOpen_Training_Folder.triggered.connect(self.open_training_folder)
        self.gui.main_view.actionOpen_Testing_Folder.triggered.connect(self.open_testing_folder)

        self.exit_action = self.gui.findChild(QtWidgets.QAction, "actionExit")
        self.exit_action.triggered.connect(partial(self.gui.closeEvent, self.exit_action))

        # create thread pool for conversion, training, plotting tasks to run in the background
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(10)
        print(f"Max thread count: {self.threadpool.maxThreadCount()}")

    def close_controller(self):
        if self.config_hdlr.write_config():
            print("Configuration has been updated")

    def show_about_message(self):
        self.gui.showInfo("About stuff", window_title="About MStat GUI v1")

    def show_model_info(self):
        if self.pcalda_ctrl.isTrained():
            _, _, meta_info = self.pcalda_ctrl.get_model()
            print(meta_info.keys())
            pcalda_str = f"Model Steps:\n{meta_info['steps']}\n\n"
            data_str = f"Data Parameters\n\nBin Size: {meta_info['bin_size']}\nTraining Data Source (by class):"
            for i in meta_info['training_files']:
                data_str += f"\n{i}\n"
                for j in meta_info['training_files'][i]:
                    #print(self.training_dict[i][j].keys())
                    data_str += f"\t{j}\n"
            cv_results = meta_info['cv_results']
            cv_str = "\n\nModel has cross validation accuracy of %0.3f +/- %0.3f" % (cv_results['test_score'].mean(), cv_results['test_score'].std())
            cv_str += """\n\tAvg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean())
            message = pcalda_str + data_str + cv_str
            self.gui.showInfo(message, "PCA-LDA Model Information")
        else:
            self.gui.showError("Model has not been trained yet.")

    def open_training_folder(self):
        #self.gui.showInfo("You clicked Open Training Folder")
        prev_dir = self.config_hdlr.get_option('train directory')
        response = self.gui.folder_dialog(prev_dir)
        print("Opening training folder", response)
        self.config_hdlr.set_option('train directory', response)
        root_index = self.trainingdir_model.setRootPath(response)
        self.gui.main_view.trainingfolder_tview.setRootIndex(root_index)

    def open_testing_folder(self):
        #self.gui.showInfo("You clicked Open Testing Folder")
        prev_dir = self.config_hdlr.get_option('test directory')
        response = self.gui.folder_dialog(prev_dir)
        print("Opening testing folder", response)
        self.config_hdlr.set_option('test directory', response)
        self.testingdir_model.setRootPath(response)
        root_index = self.testingdir_model.index(response)
        self.gui.main_view.testingfolder_tview.setRootIndex(root_index)

    def set_state(self, new_state : ControlStates):
        if new_state == ControlStates.READY:
            print("\tReady\t".center(80, '*'))
            if self.ctrl_state == ControlStates.CONVERTING:
                self.conversion_ctrl = None
            self.ctrl_state = new_state
            self.gui.reset_status_bar()
            self.gui.main_view.trainingfolder_tview.setEnabled(True)
            self.gui.main_view.testingfolder_tview.setEnabled(True)
            self.gui.main_view.plotteddata_view.setEnabled(True)
            self.update_model_data()
            return True
        elif new_state == ControlStates.CONVERTING:
            if self.ctrl_state == ControlStates.READY:
                print("\tConverting Data\t".center(80, '*'))
                self.ctrl_state = new_state
                self.gui.setStatusBarMessage("Converting Data")
                self.gui.statusbar_button.setEnabled(False)
                self.gui.main_view.trainingfolder_tview.setEnabled(False)
                self.gui.main_view.testingfolder_tview.setEnabled(False)
                return True
            return False
        elif new_state == ControlStates.BUILDING:
            if self.ctrl_state == ControlStates.READY:
                print("\tTraining PCA-LDA Model\t".center(80, '*'))
                self.ctrl_state = new_state
                self.gui.setStatusBarMessage("Training PCA-LDA Model")
                self.gui.main_view.trainingfolder_tview.setEnabled(False)
                self.gui.main_view.testingfolder_tview.setEnabled(False)
                #time.sleep(3)
                return True
            return False
        elif new_state == ControlStates.PLOTTING:
            if self.ctrl_state in [ControlStates.READY, ControlStates.LOADING, ControlStates.BUILDING]:
                print("\tPlotting\t".center(80, '*'))
                self.ctrl_state = new_state
                self.gui.setStatusBarMessage("Plotting")
                self.gui.main_view.trainingfolder_tview.setEnabled(False)
                self.gui.main_view.testingfolder_tview.setEnabled(False)
                self.update_plot_data()
                return True
            return False
        elif new_state == ControlStates.LOADING:
            if self.ctrl_state == ControlStates.READY:
                self.ctrl_state = new_state
                self.gui.setStatusBarMessage("Loading data")
                self.gui.main_view.plotteddata_view.setEnabled(False)
                return True

    def connect_plot_options(self):
        self.gui.main_view.model_combo.currentIndexChanged.connect(self.change_model_option)
        self.gui.main_view.xaxis_combo.currentIndexChanged.connect(self.update_plot_options)
        self.gui.main_view.yaxis_combo.currentIndexChanged.connect(self.update_plot_options)
        self.gui.main_view.showlegend_check.stateChanged.connect(self.update_plot_options)
        self.gui.main_view.sampleorder_check.stateChanged.connect(self.update_plot_options)
        self.gui.main_view.testdata_check.stateChanged.connect(self.update_plot_options)

    def disconnect_plot_options(self):
        with contextlib.suppress(TypeError):
            self.gui.main_view.model_combo.currentIndexChanged.disconnect()
            self.gui.main_view.xaxis_combo.currentIndexChanged.disconnect()
            self.gui.main_view.yaxis_combo.currentIndexChanged.disconnect()
            self.gui.main_view.showlegend_check.stateChanged.disconnect()
            self.gui.main_view.sampleorder_check.stateChanged.disconnect()
            self.gui.main_view.testdata_check.stateChanged.disconnect()

    def change_model_option(self):
        self.disconnect_plot_options()

        model_option = self.gui.main_view.model_combo.currentText()
        if self.pcalda_ctrl.trained_flag:
            if model_option == 'PCA Scores':
                pcs = [f'PC{i+1}' for i in range(self.pcalda_ctrl.get_num_pca_dim())]
                self.gui.set_xaxis_combo(pcs, 0)
                self.gui.set_yaxis_combo(pcs, 1)
                self.gui.main_view.xaxis_combo.setEnabled(True)
                self.gui.main_view.yaxis_combo.setEnabled(True)
            elif model_option == 'PCA-LDA Scores':
                lds = [f'LD{i+1}' for i in range(self.pcalda_ctrl.get_num_lda_dim())]
                print('lds', lds)
                self.gui.set_xaxis_combo(lds, 0)
                self.gui.set_yaxis_combo(lds, 1)
                self.gui.main_view.xaxis_combo.setEnabled(True)
                if len(lds) > 1:
                    self.gui.main_view.yaxis_combo.setEnabled(True)
                else:
                    self.gui.main_view.yaxis_combo.setEnabled(False)
            else:
                self.gui.main_view.xaxis_combo.setEnabled(False)
                self.gui.main_view.yaxis_combo.setEnabled(False)
        
        self.update_plot_options(False)

    def update_plot_options(self, disconnect=True):
        if disconnect:
            self.disconnect_plot_options()

        show_legend = (self.gui.main_view.showlegend_check.checkState() == 2)
        sample_order = (self.gui.main_view.sampleorder_check.checkState() == 2)
        show_testdata = (self.gui.main_view.testdata_check.checkState() == 2)

        model_option = self.gui.main_view.model_combo.currentText()
        xaxis_option = self.gui.main_view.xaxis_combo.currentText()
        yaxis_option = self.gui.main_view.yaxis_combo.currentText()
        
        self.connect_plot_options()

        self.plotter.change_options((show_legend, sample_order, show_testdata, model_option, xaxis_option, yaxis_option))
        self.set_state(ControlStates.PLOTTING)

    def update_plot_data(self):
        # data depends on the options chosen for plotting data (pca vs lda, which PCs or LDs, etc)
        cur_options = self.plotter.get_options()
        #print('Plot Model Option', cur_options['model_option'])

        if self.pcalda_ctrl.trained_flag:
            if cur_options['model_option'] == 'PCA Scores':
                self.plotter.clear_data()
                
                pcx = int(''.join(filter(str.isdigit, cur_options['xaxis_option'])))-1
                pcy = int(''.join(filter(str.isdigit, cur_options['yaxis_option'])))-1

                if self.pcalda_ctrl.training_dict.keys():       # need to iterate through keys and build plotting data array
                    training_tuple = self.pcalda_ctrl.getPCALDATrainingScores(False, pcx, pcy)  # also choose PCs to plot here as arguments
                    #print(f"Train Scores & Labels:\n{training_tuple[0].shape}")

                    # get training meta data HERE

                    testing_tuple = None
                    if self.pcalda_ctrl.testing_dict.keys():
                        testing_tuple = self.pcalda_ctrl.getPCALDATestingScores(False, pcx, pcy)
                        #print(f"Test Scores & Labels:\n{testing_tuple[0].shape}")

                        # get testing meta data HERE
                    
                    self.plotter.plot_pcalda_data(training_tuple, testing_tuple, title="PCA Scores")
            elif cur_options['model_option'] == 'PCA-LDA Scores':
                self.plotter.clear_data()
                print("xaxis option", cur_options['xaxis_option'])
                print("yaxis option", cur_options['yaxis_option'])
                ldx = int(''.join(filter(str.isdigit, cur_options['xaxis_option'])))-1
                if cur_options['yaxis_option'] != '':
                    ldy = int(''.join(filter(str.isdigit, cur_options['yaxis_option'])))-1
                else:
                    ldy = ldx
                #print(f"training keys: {self.pcalda_ctrl.training_dict.keys()}")
                #print(f"testing keys: {self.pcalda_ctrl.testing_dict.keys()}")
                if self.pcalda_ctrl.training_dict.keys():       # need to iterate through keys and build plotting data array
                    training_tuple = self.pcalda_ctrl.getPCALDATrainingScores(True, ldx, ldy)  # also choose PCs to plot here as arguments
                    #print(f"Train Scores & Labels:\n{training_tuple[0].shape}")

                    testing_tuple = None
                    if self.pcalda_ctrl.testing_dict.keys():
                        testing_tuple = self.pcalda_ctrl.getPCALDATestingScores(True, ldx, ldy)
                        #print(f"Test Scores & Labels:\n{testing_tuple[0].shape}")
                    
                    self.plotter.plot_pcalda_data(training_tuple, testing_tuple, title="PCA-LDA Scores")             
        else:
            print("Model has not been trained yet...")

        self.set_state(ControlStates.READY)

    def extern_plot(self):
        self.set_state(ControlStates.PLOTTING)

    def convert_RAW(self):
        print('conversion list', self.conversion_list)
        if self.set_state(ControlStates.CONVERTING):
            # hand off to conversion controller, listen for results
            self.conversion_ctrl = ConversionCtrl(self, self.gui, self.conversion_list, self.modeldata_model._data)
        else:
            print("Conversion not available right now...")
            print(f"Current state: {self.ctrl_state}")
    
    def change_bin_size(self):
        bin_size = float(self.gui.main_view.binsize_edit.text())
        self.pcalda_ctrl.set_bin_size(bin_size)

        if self.ctrl_state == ControlStates.READY:
            self.gui.setStatusBarMessage("Spectral bin size has changed. Please re-train model.")
            self.gui.setStatusBarButtonText("Train Model")
            self.gui.statusbar_button.setHidden(False)
            self.gui.statusprogress_bar.setHidden(False)
            self.gui.reattach_status_bar_button(self.train_model)

    def func(self):
        self.set_state(ControlStates.BUILDING)

    def train_model(self):
        check = self.set_state(ControlStates.BUILDING)
        
        #print(f"Training Data:\n{self.training_dict}")
        #print(f"Testing Data:\n{self.testing_dict}")
        print(f"Conversion list:\n{self.conversion_list}")
        if check:
            self.update_tt_dicts()    # update again to catch any change in class names
            if not self.pcalda_ctrl.training_dict:
                self.gui.showError("Training data not selected or not converted.")
                self.set_state(ControlStates.READY)
                return 0
            # hand off to PCA-LDA controller, listen for results
            pca_dim = self.gui.main_view.pcadim_edit.text()
            #bin_size = self.gui.main_view.binsize_edit.text()
            
            try:
                self.pcalda_ctrl.build_model(int(pca_dim))
                self.change_model_option()
                self.gui.showInfo("Model training is complete!")
                self.set_state(ControlStates.PLOTTING)
                return 1
            except ValueError as e:
                self.gui.showError("One or more of supplied values are invalid.")
                print(e)
                self.set_state(ControlStates.READY)
                return 0
            except Exception as e:
                self.gui.showError("Unknown error occurred.")
                print("Unknown error:")
                print(e)
                self.set_state(ControlStates.READY)
                return 2
        else:
            print("Training not available right now...")
            print(f"Current state: {self.ctrl_state}")

    def clear_dir_checks(self, dir_model):
        dir_model.clearData()
        dir_model.layoutChanged.emit()
        self.training_dict = {}
        self.testing_dict = {}
        if dir_model == self.trainingdir_model:     # COME UP WITH BETTER SOLUTION FOR THIS
            self.pcalda_ctrl.training_dict = {}
        else:
            self.pcalda_ctrl.testing_dict = {}
        self.update_model_data()
        self.set_state(ControlStates.PLOTTING)

    def redraw_plot(self):
        self.update_model_data()
        self.set_state(ControlStates.PLOTTING)

    def update_tt_dicts(self) -> bool:
        train_keep = []
        test_keep = []
        train_name_changed = False
        test_name_changed = False
        
        # update new selections
        for row in self.modeldata_model._data:
            if row[3] == "Train":
                #conv_check = checkMZML(str(row[0].absolute()), row[2])  # check if all raw files have been converted
                conv_check = checkNPY(str(row[0].absolute()), row[2])
                
                # if the class name is not empty and the files have been converted and the path is not in the dictionary or conversion status has changed (to completed)
                if conv_check > 0 and (row[0] not in self.training_dict or self.training_dict[row[0]]['conv_check'] != conv_check):
                    self.set_state(ControlStates.LOADING)
                    self.pcalda_ctrl.addTrainingData(row[1], str(row[0].absolute()))
                # if the path is in the training dictionary and the class name has changed
                elif row[0] in self.training_dict and (self.training_dict[row[0]]['class_name'] != row[1]):
                    result = self.pcalda_ctrl.moveTrainingData(self.training_dict[row[0]]['class_name'], row[1], str(row[0].absolute()))
                    if result == 1 and conv_check > 0:
                        self.set_state(ControlStates.LOADING)
                        self.pcalda_ctrl.addTrainingData(row[1], str(row[0].absolute()))
                    print(f"Previous class name: {self.training_dict[row[0]]['class_name']}, New class name: {row[1]}")
                    train_name_changed = True
                
                train_keep.append(row[0])
                self.training_dict[row[0]] = {
                    'class_name': row[1],
                    'num_raw': row[2],
                    'conv_check': conv_check, 
                }
            elif row[3] == "Test":
                #conv_check = checkMZML(str(row[0].absolute()), row[2])  # check if all raw files have been converted
                conv_check = checkNPY(str(row[0].absolute()), row[2])
                
                # if the class name is not empty and the files have been converted and the path is not in the dictionary or conversion status has changed (to completed)
                if conv_check > 0 and (row[0] not in self.testing_dict or self.testing_dict[row[0]]['conv_check'] != conv_check):
                    self.set_state(ControlStates.LOADING)
                    self.pcalda_ctrl.addTestingData(row[1], str(row[0].absolute()))
                # if the path is in the training dictionary and the class name has changed
                elif row[0] in self.testing_dict and (self.testing_dict[row[0]]['class_name'] != row[1]):
                    result = self.pcalda_ctrl.moveTestingData(self.testing_dict[row[0]]['class_name'], row[1], str(row[0].absolute()))
                    if result == 1 and conv_check > 0:
                        self.set_state(ControlStates.LOADING)
                        self.pcalda_ctrl.addTestingData(row[1], str(row[0].absolute()))
                    print(f"Previous class name: {self.testing_dict[row[0]]['class_name']}, New class name: {row[1]}")
                    test_name_changed = True
                
                test_keep.append(row[0])
                self.testing_dict[row[0]] = {
                    'class_name': row[1],
                    'num_raw': row[2],
                    'conv_check': conv_check
                }
        #print(f"Any new data: {new_data_check}")
        
        # delete old selections
        to_delete = set(self.training_dict.keys()).difference(train_keep)
        #print(f"items to delete: {to_delete}")
        for dir in to_delete:
            try:
                self.pcalda_ctrl.removeTrainingData(self.training_dict[dir]['class_name'], str(dir.absolute()))
                self.set_state(ControlStates.PLOTTING)
            except KeyError as e:
                pass
            try:
                self.training_dict.pop(dir)
            except KeyError as e:
                pass
        #print(f"training_dict: {self.training_dict.keys()}")

        to_delete = set(self.testing_dict.keys()).difference(test_keep)
        for dir in to_delete:
            try:
                self.pcalda_ctrl.removeTestingData(self.testing_dict[dir]['class_name'], str(dir.absolute()))
                self.set_state(ControlStates.PLOTTING)
            except KeyError as e:
                pass
            try:
                self.testing_dict.pop(dir)
            except KeyError as e:
                pass

        return train_name_changed, test_name_changed
        
    def update_conversion_list(self):
        self.conversion_list = []
        self.conversion_list = [key for key in self.training_dict if self.training_dict[key]['conv_check'] == 0]
        self.conversion_list.extend(key for key in self.testing_dict if self.testing_dict[key]['conv_check'] == 0)

        # if some RAW files remain unconverted, offer the user the option to convert them
        if self.conversion_list and self.ctrl_state != ControlStates.LOADING:
            self.gui.setStatusBarMessage("Some selected folders do not contain converted files")
            self.gui.setStatusBarButtonText("Run File Conversion")
            self.gui.statusbar_button.setHidden(False)
            self.gui.statusprogress_bar.setHidden(False)
            self.gui.reattach_status_bar_button(self.convert_RAW)

    def update_model_data(self):
        # get the current data in the table
        #print(f"Model Data:\n{self.modeldata_model._data}")
        #print(f"Training Checks:\n{self.trainingdir_model.checks}")
        #print(f"Testing Checks:\n{self.testingdir_model.checks}")
        self.model_data_src = []
        
        # update the train data in the table
        old_display_data = [row for row in self.modeldata_model._data if row[3] == "Train"]
        popd_training_paths = [i for i in self.training_paths]
        self.training_paths = []
        for key in self.trainingdir_model.checks:
            path = self.trainingdir_model.filePath(key)
            if path in popd_training_paths:
                # keep old class name if this data has already been selected
                #print(f"PRE Train {path}")
                self.model_data_src.append(old_display_data[popd_training_paths.index(path)])
            else:
                # add new display data
                #print(f"NEW Train {path}")
                dirpath, dir = os.path.split(path)
                dirpath, parent_dir = os.path.split(dirpath)
                num_raw = get_num_files(path, '.raw')#len(glob.glob1(path, "*.raw"))
                #print('RAW files', glob.glob1(path, "*.raw"), num_raw)
                self.model_data_src.append([Path(path), f'{parent_dir}/{dir}', num_raw, "Train"]) #f'{parent_dir}/{dir}'
            self.training_paths.append(path)

        # update the test data in the table        
        old_display_data = [row for row in self.modeldata_model._data if row[3] == "Test"]
        popd_testing_paths = [i for i in self.testing_paths]
        self.testing_paths = []
        for key in self.testingdir_model.checks:
            path = self.testingdir_model.filePath(key)
            if path in popd_testing_paths:
                # keep old class name if this data has already been selected
                #print(f"PRE Test {path}")
                self.model_data_src.append(old_display_data[popd_testing_paths.index(path)])
            else:
                # add new display data
                #print(f"NEW Test {path}")
                dirpath, dir = os.path.split(path)
                dirpath, parent_dir = os.path.split(dirpath)
                num_raw = get_num_files(path, '.raw')#len(glob.glob1(path, "*.raw"))
                self.model_data_src.append([Path(path), f'{parent_dir}/{dir}', num_raw, "Test"])
            self.testing_paths.append(path)
        
        # update the table model
        self.modeldata_model.update_data(self.model_data_src)
        self.modeldata_model.define_col_flags(1, 1)
        self.modeldata_model.layoutChanged.emit()

        # update the data to be used for conversion, plotting, and training
        self.update_conversion_list()
        train_name_changed, test_name_changed = self.update_tt_dicts()

        # if there are no items to convert and we are not loading data then the status bar should be reset
        if not self.conversion_list and self.ctrl_state != ControlStates.LOADING and self.pcalda_ctrl.isTrained():
            if train_name_changed:
                self.gui.setStatusBarMessage("Some training class names have changed.")
                self.gui.setStatusBarButtonText("Retrain model")
                self.gui.statusbar_button.setHidden(False)
                self.gui.statusprogress_bar.setHidden(False)
                self.gui.reattach_status_bar_button(self.train_model)
            elif test_name_changed:
                self.gui.setStatusBarMessage("Some test class names have changed.")
                self.gui.setStatusBarButtonText("Redraw Plot")
                self.gui.statusbar_button.setHidden(False)
                self.gui.statusprogress_bar.setHidden(False)
                self.gui.reattach_status_bar_button(self.redraw_plot)
            else:
                self.gui.reset_status_bar()

import configparser as cp
from logging import config
from os.path import exists
from datetime import datetime
import sys

class ConfigHandler:
    section_name = 'MAIN'
    new_config = False
    config_created = False

    def __init__(self, config_name="mstat_config.ini", my_path = ".") -> None:
        self.config = cp.RawConfigParser()
        self.config_name = config_name
        self.my_path = my_path

    def read_config(self) -> bool:
        if exists(self.config_name):
            self.config.read(self.config_name)
            self.config_created = True
            return True
        return False

    def write_config(self) -> bool:
        if self.config_created:
            with open(self.config_name, 'w') as config_file:
                self.config.write(config_file)
            return True
        return False

    def set_option(self, option_name : str, value) -> None:
        if type(value) is str:
            self.config.set(self.section_name, option_name, value)
        else:
            self.config.set(self.section_name, option_name, str(value))
    
    def get_option(self, option_name : str, value='no val') -> str:
        return self.config.get(self.section_name, option_name, fallback=value)

    def create_config(self) -> None:
        self.config.add_section(self.section_name)
        self.config.set(self.section_name, 'last start time', str(datetime.now()))
        self.config.set(self.section_name, 'train directory', self.my_path)
        self.config.set(self.section_name, 'test directory', self.my_path)
        self.config.set(self.section_name, 'windows num bits', '64' if sys.maxsize > 2**32 else '32')
        self.config.set(self.section_name, 'msfr installed', str(False))
        self.config.set(self.section_name, 'mscv directory', '')

        self.config_created = True
        self.new_config = True   


if __name__ == "__main__":
    # check for known paths (later...)
    path = '.'#'C:/Users/Jackson/PSI Files Dropbox/MS Detectors'
    config_hdlr = ConfigHandler()
    if not config_hdlr.read_config():
        config_hdlr.create_config()


    # open the gui
    app = QtWidgets.QApplication(sys.argv)
    mstat = MSTAT_Ctrl(app, root_path=path, config_hdlr=config_hdlr)

    sys.exit(app.exec_())

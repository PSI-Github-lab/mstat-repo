


try:
    import os
    import contextlib
    from mstat.dependencies.helper_funcs import *
    from pathlib import Path
    from mstat.gui.conversion import ConversionCtrl
    from mstat.gui.data_options import * #DataOptionsCtrl
    from mstat.gui.main_gui import MetaExploreGUI
    from mstat.gui.diag_power import DiagPowerCtrl
    from mstat.gui.data_quality import DataQualityCtrl
    from mstat.gui.hier_construct import HierCtrl
    from mstat.gui.table_model import MetaTableModel
    from mstat.dependencies.helper_funcs import *
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def update_data_dicts(self, row, role : DataRole) -> bool:
    names_changed = False
    conv_check, _ = checkNPY(str(row[0].absolute()), row[2])

    if role == DataRole.TRAINING:
        ref_dict = self.training_dict
    elif role == DataRole.TESTING:
        ref_dict = self.testing_dict

    # if the class name is not empty and the files have been converted and the path is not in the dictionary or conversion status has changed (to completed)
    if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND] and (row[0] not in ref_dict or ref_dict[row[0]]['conv_check'] != conv_check):
        self.set_state(ControlStates.LOADING)
        self.pcalda_ctrl.add_data(row[1], str(row[0].absolute()), role)
    # if the path is in the training dictionary and the class name has changed
    elif row[0] in ref_dict and (ref_dict[row[0]]['class_name'] != row[1]):
        result = self.pcalda_ctrl.move_data(ref_dict[row[0]]['class_name'], row[1], str(row[0].absolute()), role)
        if result == CompFlag.FAILURE and conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
            self.set_state(ControlStates.LOADING)
            self.pcalda_ctrl.add_data(row[1], str(row[0].absolute()), role)
        print(f"Previous class name: {ref_dict[row[0]]['class_name']}, New class name: {row[1]}")
        names_changed = True
    
    if role == DataRole.TRAINING:
        self.training_dict[row[0]] = {
            'class_name': row[1],
            'num_raw': row[2],
            'conv_check': conv_check, 
        }
    elif role == DataRole.TESTING:
        self.testing_dict[row[0]] = {
            'class_name': row[1],
            'num_raw': row[2],
            'conv_check': conv_check, 
        }

    return names_changed

def update_tt_dicts(self):
    """
    update data dictionaries in the mstat cotnroller and in the model controller
    """
    #print("update_tt_dicts")
    train_keep = []
    test_keep = []
    training_names = []
    test_name_changed = False

    # update new selections
    for row in self.table_data_model._data:
        if row[3].lower() == "train":
            role = DataRole.TRAINING
            train_keep.append(row[0])
            training_names.append(row[1])
            self.update_data_dicts(row, role)
        elif row[3].lower() == "test":
            role = DataRole.TESTING
            test_keep.append(row[0])
            test_name_changed = self.update_data_dicts(row, role)

    # delete old selections
    to_delete = set(self.training_dict.keys()).difference(train_keep)
    #print(f"items to delete: {to_delete}")
    for dir in to_delete:
        with contextlib.suppress(KeyError):
            self.pcalda_ctrl.remove_data(self.training_dict[dir]['class_name'], str(dir.absolute()), DataRole.TRAINING)
            self.set_state(ControlStates.PLOTTING)
        with contextlib.suppress(KeyError):
            self.training_dict.pop(dir)
    #print(f"training_dict: {self.training_dict.keys()}")

    to_delete = set(self.testing_dict.keys()).difference(test_keep)
    for dir in to_delete:
        with contextlib.suppress(KeyError):
            self.pcalda_ctrl.remove_data(self.testing_dict[dir]['class_name'], str(dir.absolute()), DataRole.TESTING)
            self.set_state(ControlStates.PLOTTING)
        with contextlib.suppress(KeyError):
            self.testing_dict.pop(dir)
    return training_names, test_name_changed
    
def update_conversion_list(self):
    #print("update_conversion_list")
    self.conversion_list = []
    self.conversion_list = [key for key in self.training_dict if self.training_dict[key]['conv_check'] not in [CompFlag.SUCCESS, CompFlag.DATAFND]]
    self.conversion_list.extend(key for key in self.testing_dict if self.testing_dict[key]['conv_check'] not in [CompFlag.SUCCESS, CompFlag.DATAFND])

    # if some RAW files remain unconverted, offer the user the option to convert them
    if self.conversion_list and self.ctrl_state != ControlStates.LOADING:
        self.gui.setStatusBarMessage("Some selected folders do not contain converted files")
        self.gui.setStatusBarButtonText("Run File Conversion")
        self.gui.statusbar_button.setHidden(False)
        self.gui.statusprogress_bar.setHidden(False)
        self.gui.reattach_status_bar_button(self.convert_RAW)

def update_training_table_data(self):
    # update the train data in the tables
    old_display_data = [row for row in self.table_data_model._data if row[3] == "Train"]
    popd_training_paths = list(self.training_paths)
    self.training_paths = []
    # iterate through all checked directories
    for key in self.trainingdir_model.checks:
        path = self.trainingdir_model.filePath(key)
        if path in popd_training_paths:
            # keep old class name if this data has already been selected
            self.model_data_src.append(old_display_data[popd_training_paths.index(path)])
        else:
            # add new data to the table
            dirpath, dir = os.path.split(path)
            dirpath, parent_dir = os.path.split(dirpath)
            num_files = get_num_files(path, '.raw')#len(glob.glob1(path, "*.raw"))
            class_suggestion = f'{parent_dir}/{dir}'
            conv_check, num_rows = checkNPY(path, num_files)
            if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
                num_files = num_rows
                with open(npy_file_name(path), 'rb') as f:
                    _ = np.load(f, allow_pickle=True)
                    _ = np.load(f, allow_pickle=True)
                    meta = np.load(f, allow_pickle=True)
                if meta[0]['comment1'] != '':
                    class_suggestion = meta[0]['comment1']
            #print('RAW files', glob.glob1(path, "*.raw"), num_raw)
            self.model_data_src.append([Path(path), class_suggestion, num_files, "Train"]) #f'{parent_dir}/{dir}'
        self.training_paths.append(path)

def update_testing_table_data(self):
    # update the test data in the tables     
    old_display_data = [row for row in self.table_data_model._data if row[3] == "Test"]
    popd_testing_paths = list(self.testing_paths)
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
            num_files = get_num_files(path, '.raw')#len(glob.glob1(path, "*.raw"))
            class_suggestion = f'{parent_dir}/{dir}'
            conv_check, num_rows = checkNPY(path, num_files)
            if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
                num_files = num_rows
                with open(npy_file_name(path), 'rb') as f:
                    _ = np.load(f, allow_pickle=True)
                    _ = np.load(f, allow_pickle=True)
                    meta = np.load(f, allow_pickle=True)
                if meta[0]['comment1'] != '':
                    class_suggestion = meta[0]['comment1']
            self.model_data_src.append([Path(path), class_suggestion, num_files, "Test"])
        self.testing_paths.append(path)

def update_model_data(self):
    """
    Update the model table data to include/remove checked & unchecked data
    """
    #print("update_model_data")
    # get the current data in the table
    self.model_data_src = []
    
    self.update_training_table_data()
    
    self.update_testing_table_data()
    
    # update the table model
    self.table_data_model.update_data(self.model_data_src)
    self.table_data_model.define_col_flags(1, 1)
    self.table_data_model.layoutChanged.emit()

    # update the data to be used for conversion, plotting, and training
    self.update_conversion_list()
    class_names, test_name_changed = self.update_tt_dicts()

    # if there are no items to convert and we are not loading data then the status bar should be reset
    #self.gui.reset_status_bar()
    #if not self.conversion_list and self.ctrl_state != ControlStates.LOADING and self.pcalda_ctrl.isTrained():
    if not self.conversion_list:
        self.gui.reset_status_bar()
    
    print(class_names, self.pcalda_ctrl.le_classes)
    if self.pcalda_ctrl.isTrained() and any_member_not_present(class_names, self.pcalda_ctrl.le_classes) and np.array(class_names).shape[0] == np.array(self.pcalda_ctrl.le_classes).shape[0]:
        #print(np.array(class_names).shape[0], np.array(self.pcalda_ctrl.le_classes).shape[0])
        self.gui.setStatusBarMessage("Some training class names have changed.")
        self.gui.setStatusBarButtonText("Retrain model")
        self.gui.statusbar_button.setHidden(False)
        self.gui.statusprogress_bar.setHidden(False)
        self.gui.reattach_status_bar_button(self.start_build_model)
    elif test_name_changed:
        self.gui.setStatusBarMessage("Some test class names have changed.")
        self.gui.setStatusBarButtonText("Redraw Plot")
        self.gui.statusbar_button.setHidden(False)
        self.gui.statusprogress_bar.setHidden(False)
        self.gui.reattach_status_bar_button(self.redraw_plot)

def convert_RAW(self, include_everything=False):
    # check first row for empty table
    first_row = self.table_data_model._data[0]
    if first_row[0] != '':
        if include_everything:
            self.conversion_list = [model_row[0] for model_row in self.table_data_model._data]

        print('conversion list', self.conversion_list)
        if self.set_state(ControlStates.CONVERTING):
            # hand off to conversion controller, listen for results
            self.conversion_ctrl = ConversionCtrl(self, self.gui, self.conversion_list, self.table_data_model._data)
        else:
            print("Conversion not available right now...")
            print(f"Current state: {self.ctrl_state}")

def open_data_options(self):
    self.data_option_ctrl = DataOptionsCtrl(self, self.pcalda_ctrl, self.gui)

def hier_clustering(self):
    training_keys = self.pcalda_ctrl.training_dict.keys()
    
    if self.pcalda_ctrl.isTrained():
        if len(training_keys) >= 2:
            self.hier_ctrl = HierCtrl(self, self.pcalda_ctrl, self.gui)
        else:
            self.gui.showError("Need at least two classes to perform analysis.")
    else:
        self.gui.showError("Train model before hierarchical clustering.")

def diag_power_analysis(self):
    training_keys = self.pcalda_ctrl.training_dict.keys()
    
    if self.pcalda_ctrl.isTrained():
        if len(training_keys) >= 2:
            self.diag_power_ctrl = DiagPowerCtrl(self, self.pcalda_ctrl, self.gui)
        else:
            self.gui.showError("Need at least two classes to perform analysis.")
    else:
        self.gui.showError("Train model before diagnostic power analysis.")

def data_quality_analysis(self):
    training_keys = self.pcalda_ctrl.training_dict.keys()
    for key in training_keys:
        print(key, self.pcalda_ctrl.training_dict[key].keys())
    
    if len(training_keys) >= 1:
        self.data_quality_ctrl = DataQualityCtrl(self, self.pcalda_ctrl, self.gui)
    else:
        self.gui.showError("No data to perform analysis.")

def explore_meta(self):
    table_data = self.table_data_model.get_data()

    try:
        # go through paths and only consider those that have been converted to NPY format
        npy_file_names = []
        for data_row in table_data:
            path = str(data_row[0].absolute())
            conv_check, _ = checkNPY(path, data_row[2])
            if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
                npy_file_names.append(npy_file_name(path))

        print(npy_file_names)

        meta_header = []
        metadata = []
        for file in npy_file_names:
            with open(file, 'rb') as f:
                _ = np.load(f, allow_pickle=True)
                _ = np.load(f, allow_pickle=True)
                meta = np.load(f, allow_pickle=True)

                # get metadata item names
                if not meta_header:
                    meta_header = list(meta[0])

                for meta_dict in meta:
                    meta_row = []
                    for key in meta_header:
                        try:
                            meta_row.append(meta_dict[key])
                        except KeyError:
                            meta_row.append('N/A')
                    metadata.append(meta_row)

        self.meta_explore = MetaExploreGUI(self)

        # populate the table headers with meta data items
        #meta_table = self.meta_explore.dialog_view.metadata_table
        #meta_table.setColumnCount(len(meta_header))
        #meta_table.setHorizontalHeaderLabels(meta_header)

        meta_model = MetaTableModel(self, meta_header, metadata)
        self.meta_explore.set_up_table(meta_model, meta_header)

        self.meta_explore.show()
    except AttributeError as exc:
        print(f'From {os.path.basename(__file__)}')
        print(exc)
        self.gui.showError("No data to explore.")
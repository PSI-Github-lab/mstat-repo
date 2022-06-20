try:
    import os
    from mstat.dependencies.helper_funcs import *
    from pathlib import Path
    from mstat.gui.conversion import ConversionCtrl
    from mstat.gui.main_gui import MetaExploreGUI
    from mstat.gui.table_model import MetaTableModel
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def update_tt_dicts(self) -> bool:
    train_keep = []
    test_keep = []
    train_name_changed = False
    test_name_changed = False
    
    # update new selections
    for row in self.table_data_model._data:
        if row[3] == "Train":
            #conv_check = checkMZML(str(row[0].absolute()), row[2])  # check if all raw files have been converted
            conv_check, _ = checkNPY(str(row[0].absolute()), row[2])
            print('conv_check', conv_check)
            
            # if the class name is not empty and the files have been converted and the path is not in the dictionary or conversion status has changed (to completed)
            if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND] and (row[0] not in self.training_dict or self.training_dict[row[0]]['conv_check'] != conv_check):
                self.set_state(ControlStates.LOADING)
                self.pcalda_ctrl.addTrainingData(row[1], str(row[0].absolute()))
            # if the path is in the training dictionary and the class name has changed
            elif row[0] in self.training_dict and (self.training_dict[row[0]]['class_name'] != row[1]):
                result = self.pcalda_ctrl.moveTrainingData(self.training_dict[row[0]]['class_name'], row[1], str(row[0].absolute()))
                if result == CompFlag.FAILURE and conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
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
            conv_check, _ = checkNPY(str(row[0].absolute()), row[2])
            
            # if the class name is not empty and the files have been converted and the path is not in the dictionary or conversion status has changed (to completed)
            if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND] and (row[0] not in self.testing_dict or self.testing_dict[row[0]]['conv_check'] != conv_check):
                self.set_state(ControlStates.LOADING)
                self.pcalda_ctrl.addTestingData(row[1], str(row[0].absolute()))
            # if the path is in the training dictionary and the class name has changed
            elif row[0] in self.testing_dict and (self.testing_dict[row[0]]['class_name'] != row[1]):
                result = self.pcalda_ctrl.moveTestingData(self.testing_dict[row[0]]['class_name'], row[1], str(row[0].absolute()))
                if result == CompFlag.FAILURE and conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
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
    self.conversion_list = [key for key in self.training_dict if self.training_dict[key]['conv_check'] not in [CompFlag.SUCCESS, CompFlag.DATAFND]]
    self.conversion_list.extend(key for key in self.testing_dict if self.testing_dict[key]['conv_check'] not in [CompFlag.SUCCESS, CompFlag.DATAFND])

    # if some RAW files remain unconverted, offer the user the option to convert them
    if self.conversion_list and self.ctrl_state != ControlStates.LOADING:
        self.gui.setStatusBarMessage("Some selected folders do not contain converted files")
        self.gui.setStatusBarButtonText("Run File Conversion")
        self.gui.statusbar_button.setHidden(False)
        self.gui.statusprogress_bar.setHidden(False)
        self.gui.reattach_status_bar_button(self.convert_RAW)

def update_model_data(self):
    """
    Update the model table data to include/remove checked & unchecked data
    """
    # get the current data in the table
    self.model_data_src = []
    
    # update the train data in the tables
    old_display_data = [row for row in self.table_data_model._data if row[3] == "Train"]
    popd_training_paths = [i for i in self.training_paths]
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
                if meta[0]['comment1'] is not '':
                    class_suggestion = meta[0]['comment1']
            #print('RAW files', glob.glob1(path, "*.raw"), num_raw)
            self.model_data_src.append([Path(path), class_suggestion, num_files, "Train"]) #f'{parent_dir}/{dir}'
        self.training_paths.append(path)

    # update the test data in the tables     
    old_display_data = [row for row in self.table_data_model._data if row[3] == "Test"]
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
            num_files = get_num_files(path, '.raw')#len(glob.glob1(path, "*.raw"))
            class_suggestion = f'{parent_dir}/{dir}'
            conv_check, num_rows = checkNPY(path, num_files)
            if conv_check in [CompFlag.SUCCESS, CompFlag.DATAFND]:
                num_files = num_rows
                with open(npy_file_name(path), 'rb') as f:
                    _ = np.load(f, allow_pickle=True)
                    _ = np.load(f, allow_pickle=True)
                    meta = np.load(f, allow_pickle=True)
                if meta[0]['comment1'] is not '':
                    class_suggestion = meta[0]['comment1']
            self.model_data_src.append([Path(path), class_suggestion, num_files, "Test"])
        self.testing_paths.append(path)
    
    # update the table model
    self.table_data_model.update_data(self.model_data_src)
    self.table_data_model.define_col_flags(1, 1)
    self.table_data_model.layoutChanged.emit()

    # update the data to be used for conversion, plotting, and training
    self.update_conversion_list()
    train_name_changed, test_name_changed = self.update_tt_dicts()

    # if there are no items to convert and we are not loading data then the status bar should be reset
    #self.gui.reset_status_bar()
    #if not self.conversion_list and self.ctrl_state != ControlStates.LOADING and self.pcalda_ctrl.isTrained():
    if not self.conversion_list:
        self.gui.reset_status_bar()
    
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

def convert_RAW(self):
        print('conversion list', self.conversion_list)
        if self.set_state(ControlStates.CONVERTING):
            # hand off to conversion controller, listen for results
            self.conversion_ctrl = ConversionCtrl(self, self.gui, self.conversion_list, self.table_data_model._data)
        else:
            print("Conversion not available right now...")
            print(f"Current state: {self.ctrl_state}")

def explore_meta(self):
    table_data = self.table_data_model.get_data()
    
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
                meta_header = [key for key in meta[0]]

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
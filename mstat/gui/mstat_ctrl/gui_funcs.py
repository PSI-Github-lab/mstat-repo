try:
    import os
    from mstat.dependencies.helper_funcs import *
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

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
        if cv_results == CompFlag.FAILURE:
            cv_str = "\n\nNo cross validation performed"
        else:
            cv_str = "\n\nModel has cross validation accuracy of %0.3f +/- %0.3f" % (cv_results['test_score'].mean(), cv_results['test_score'].std())
        cv_str += """\n\tAvg fit time of %0.4f and score time of %0.4f""" % (cv_results['fit_time'].mean(), cv_results['score_time'].mean())
        message = pcalda_str + data_str + cv_str
        self.gui.showInfo(message, "PCA-LDA Model Information")
    else:
        self.gui.showError("Model has not been trained yet.")

def open_training_folder(self):
    #self.gui.showInfo("You clicked Open Training Folder")
    prev_dir = self.config_hdlr.get_option('MAIN', 'train directory')
    response = self.gui.folder_dialog(prev_dir)
    if response != '':
        print("Opening training folder", response)
        self.config_hdlr.set_option('MAIN', 'train directory', response)
        root_index = self.trainingdir_model.setRootPath(response)
        self.gui.main_view.trainingfolder_tview.setRootIndex(root_index)
    else:
        print("Action cancelled")

def open_testing_folder(self):
    #self.gui.showInfo("You clicked Open Testing Folder")
    prev_dir = self.config_hdlr.get_option('MAIN', 'test directory')
    response = self.gui.folder_dialog(prev_dir)
    if response != '':
        print("Opening testing folder", response)
        self.config_hdlr.set_option('MAIN', 'test directory', response)
        root_index = self.testingdir_model.setRootPath(response)
        self.gui.main_view.testingfolder_tview.setRootIndex(root_index)

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
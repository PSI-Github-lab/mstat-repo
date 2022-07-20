try:
    import os
    import time
    from mstat.dependencies.helper_funcs import *
    from mstat.gui.binning import BinningCtrl
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def change_bins(self):
    self.binning_ctrl = BinningCtrl(self, self.gui, self.pcalda_ctrl)
    
def update_bins(self, bin_size, low_lim, up_lim):
    self.pcalda_ctrl.bin_size, self.pcalda_ctrl.low_lim, self.pcalda_ctrl.up_lim = bin_size, low_lim, up_lim
    if self.ctrl_state == ControlStates.READY and self.pcalda_ctrl.isTrained():
        self.gui.setStatusBarMessage("Spectral bin options have changed. Please re-train model to update bins.")
        self.gui.setStatusBarButtonText("Train Model")
        self.gui.statusbar_button.setHidden(False)
        self.gui.statusprogress_bar.setHidden(False)
        self.gui.reattach_status_bar_button(self.start_build_model)

def start_build_model(self):
    check = self.set_state(ControlStates.BUILDING)
    
    #print(f"Training Data:\n{self.training_dict}")
    #print(f"Testing Data:\n{self.testing_dict}")
    print(f"Conversion list:\n{self.conversion_list}")
    if check:
        self.update_tt_dicts()    # update again to catch any change in class names
        if not self.pcalda_ctrl.training_dict:
            self.gui.showError("Training data not selected or not converted.")
            self.set_state(ControlStates.READY)
        else:
            # hand off to PCA-LDA controller, listen for results
            pca_dim = self.gui.main_view.pcadim_spin.value()
            try:
                #start_time = time.time()
                do_diff = (self.data_options['perform differentiation'] == 'True')
                diff_order = int(self.data_options['differentiation order'])
                self.pcalda_ctrl.build_model(int(pca_dim), do_diff, diff_order)
                #print(f"--- completed in {time.time() - start_time} seconds ---")
                #print(model_exceptions[0])
                
            except ValueError as e:
                self.gui.showError(f"One or more of supplied values are invalid.\n{e}")
                print(e)
                print(f'From {os.path.basename(__file__)}')
                self.set_state(ControlStates.READY)
            except Exception as e:
                self.gui.showError(f"Unknown error occurred:\n{e}")
                print("Unknown error:")
                print(e)
                print(f'From {os.path.basename(__file__)}')
                self.set_state(ControlStates.READY)
    else:
        print("Training not available right now...")
        print(f"Current state: {self.ctrl_state}")

def start_learning_curve(self):
    if self.pcalda_ctrl.isTrained():
        self.pcalda_ctrl.learning_curve()

def start_test_model(self):
    self.set_state(ControlStates.BUILDING)
    self.pcalda_ctrl.test_model()

def start_test_single_file(self):
    self.pcalda_ctrl.test_single_file()

def delete_model(self):
    if self.pcalda_ctrl.isTrained():
        self.pcalda_ctrl.reset_model()
        self.reset_plot()
        self.gui.showInfo("Model has been deleted.")
    else:
        print("No model to delete")

def save_model(self):
    if self.pcalda_ctrl.isTrained():
        name, opt = self.gui.show_file_save('Save Model', self.root_path)
        if name != '':
            self.pcalda_ctrl.save_model(name)
    else:
        self.gui.showError('Must train model before saving.')

def open_model(self):
    file_name = self.gui.file_dialog(self.main_training_dir, dialog_caption='Select a model file', type_filter="MODEL Files (*.model)")

    if file_name != '':
        self.pcalda_ctrl.load_model(file_name)
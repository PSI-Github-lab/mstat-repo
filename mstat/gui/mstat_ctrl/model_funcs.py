try:
    import os
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
        self.gui.reattach_status_bar_button(self.train_model)

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
            return CompFlag.FAILURE
        # hand off to PCA-LDA controller, listen for results
        pca_dim = self.gui.main_view.pcadim_edit.text()
        
        try:
            model_exceptions = self.pcalda_ctrl.build_model(int(pca_dim))
            print(model_exceptions[0])
            if model_exceptions[0] is None:
                self.change_model_option()
                self.gui.showInfo("Model training is complete!")
                self.set_state(ControlStates.PLOTTING)
                return CompFlag.SUCCESS
            else:
                self.gui.showInfo(f"Model training is completed with the following exceptions:\n{model_exceptions}")
                self.set_state(ControlStates.READY)
                return CompFlag.FAILURE
        except ValueError as e:
            self.gui.showError(f"One or more of supplied values are invalid.\n{e}")
            print(e)
            self.set_state(ControlStates.READY)
            return CompFlag.FAILURE
        except Exception as e:
            self.gui.showError(f"Unknown error occurred:\n{e}")
            print("Unknown error:")
            print(e)
            self.set_state(ControlStates.READY)
            return CompFlag.UNKNERR
    else:
        print("Training not available right now...")
        print(f"Current state: {self.ctrl_state}")

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
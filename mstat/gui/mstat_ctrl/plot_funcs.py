try:
    import os
    import contextlib
    from mstat.dependencies.helper_funcs import *
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def connect_plot_options(self):
    self.gui.main_view.model_combo.currentIndexChanged.connect(self.change_model_option)
    self.gui.main_view.xaxis_combo.currentIndexChanged.connect(self.update_plot_options)
    self.gui.main_view.yaxis_combo.currentIndexChanged.connect(self.update_plot_options)
    self.gui.main_view.showlegend_check.stateChanged.connect(self.update_plot_options)
    self.gui.main_view.sampleorder_check.stateChanged.connect(self.update_plot_options)
    self.gui.main_view.testlabel_check.stateChanged.connect(self.update_plot_options)

def disconnect_plot_options(self):
    with contextlib.suppress(TypeError):
        self.gui.main_view.model_combo.currentIndexChanged.disconnect()
        self.gui.main_view.xaxis_combo.currentIndexChanged.disconnect()
        self.gui.main_view.yaxis_combo.currentIndexChanged.disconnect()
        self.gui.main_view.showlegend_check.stateChanged.disconnect()
        self.gui.main_view.sampleorder_check.stateChanged.disconnect()
        self.gui.main_view.testlabel_check.stateChanged.disconnect()

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
    show_testdata = (self.gui.main_view.testlabel_check.checkState() == 2)

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

def redraw_plot(self):
    self.update_model_data()
    self.set_state(ControlStates.PLOTTING)

def reset_plot(self):
    self.plotter.setup_plot()
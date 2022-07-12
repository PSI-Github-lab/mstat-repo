try:
    import os
    import contextlib
    from PyQt5 import QtCore
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
            if self.pcalda_ctrl.get_num_classes() > 1:
                lds = [f'LD{i+1}' for i in range(self.pcalda_ctrl.get_num_lda_dim())]
                self.gui.set_xaxis_combo(lds, 0)
                self.gui.set_yaxis_combo(lds, 1)
                self.gui.main_view.xaxis_combo.setEnabled(True)
                if len(lds) > 1:
                    self.gui.main_view.yaxis_combo.setEnabled(True)
                else:
                    self.gui.main_view.yaxis_combo.setEnabled(False)
            else:
                self.gui.showError("Only PCA is available for model with one class of data.")
                self.gui.main_view.model_combo.setCurrentIndex(0)
                #self.connect_plot_options()
                #return
        elif model_option == 'PCA Loadings':
            pcs = [f'PC{i+1}' for i in range(self.pcalda_ctrl.get_num_pca_dim())]
            self.gui.set_xaxis_combo(pcs, 0)
            self.gui.set_yaxis_combo(pcs, 0)
            self.gui.main_view.xaxis_combo.setEnabled(False)
            self.gui.main_view.yaxis_combo.setEnabled(True)
        elif model_option == 'PCA-LDA Loadings':
            if self.pcalda_ctrl.get_num_classes() > 1:
                lds = [f'LD{i+1}' for i in range(self.pcalda_ctrl.get_num_lda_dim())]
                self.gui.set_xaxis_combo(lds, 0)
                self.gui.set_yaxis_combo(lds, 0)
                self.gui.main_view.xaxis_combo.setEnabled(False)
                self.gui.main_view.yaxis_combo.setEnabled(True)
            else:
                self.gui.showError("Only PCA is available for model with one class of data.")
                self.gui.main_view.model_combo.setCurrentIndex(2)
                #self.connect_plot_options()
                #return
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

class PlotWorkerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal()
    progressComplete = QtCore.pyqtSignal(tuple, tuple, str, list, bool)

class PlotWorker(QtCore.QRunnable):
    def __init__(self, pcalda_ctrl, main_option, ldx, ldy, label_test, external):
        super(PlotWorker, self).__init__()
        self.signals = PlotWorkerSignals()
        self.pcalda_ctrl = pcalda_ctrl
        self.main_option = main_option
        self.ldx, self.ldy, self.label_test = ldx, ldy, label_test
        self.external = external

    def run(self):
        if self.pcalda_ctrl.training_dict.keys():
            training_tuple = self.pcalda_ctrl.getPCALDATrainingScores(self.main_option == 'PCA-LDA Scores', self.ldx, self.ldy, self.label_test)  # also choose PCs to plot here as arguments
            #print(f"Train Scores & Labels:\n{training_tuple[0].shape}")
        else:
            training_tuple = (None, None, None, None, None, None)
        
        if self.pcalda_ctrl.testing_dict.keys():
            testing_tuple = self.pcalda_ctrl.getPCALDATestingScores(self.main_option == 'PCA-LDA Scores', self.ldx, self.ldy, self.label_test)
        else:
            testing_tuple = (None, None, None, None, None, None)

        self.signals.progressComplete.emit(training_tuple, testing_tuple, self.main_option, self.pcalda_ctrl.le_classes, self.external)   

def on_plot_complete(self, training_tuple, testing_tuple, title, le_classes, external):
    if not external:
        self.plotter.clear_data()
        self.plotter.plot_pcalda_data(training_tuple, testing_tuple, title=title, model_classes=le_classes)
    else:
        self.plotter.ext_plot_pcalda_data(training_tuple, testing_tuple, title=title, model_classes=le_classes)
    self.set_state(ControlStates.READY)

def plot_data(self, external=False):
    # data depends on the options chosen for plotting data (pca vs lda, which PCs or LDs, etc)
    cur_options = self.plotter.get_options()
    #print('Plot Model Option', cur_options['model_option'])
    main_option = cur_options['model_option'] 

    if self.pcalda_ctrl.trained_flag:
        if main_option in ['PCA Scores', 'PCA-LDA Scores']:
            #print("xaxis option", cur_options['xaxis_option'])
            #print("yaxis option", cur_options['yaxis_option'])
            ldx = int(''.join(filter(str.isdigit, cur_options['xaxis_option'])))-1
            ldy = int(''.join(filter(str.isdigit, cur_options['yaxis_option']))) - 1 if cur_options['yaxis_option'] != '' else ldx

            label_test = cur_options['show_test_checked']

            worker = PlotWorker(self.pcalda_ctrl, main_option, ldx, ldy, label_test, external)
            worker.setAutoDelete(True)
            #worker.signals.progressChanged.connect(self.on_data_worker_update)
            worker.signals.progressComplete.connect(self.on_plot_complete)
            #print(f"\tMaking frame for {new_mzml_path}...")
            self.threadpool.start(worker)

            '''if self.pcalda_ctrl.training_dict.keys():
                training_tuple = self.pcalda_ctrl.getPCALDATrainingScores(main_option == 'PCA-LDA Scores', ldx, ldy, label_test)  # also choose PCs to plot here as arguments
                #print(f"Train Scores & Labels:\n{training_tuple[0].shape}")
            else:
                training_tuple = None
            
            if self.pcalda_ctrl.testing_dict.keys():
                testing_tuple = self.pcalda_ctrl.getPCALDATestingScores(main_option == 'PCA-LDA Scores', ldx, ldy, label_test)
            else:
                testing_tuple = None
            
            if not external:
                self.plotter.clear_data()
                self.plotter.plot_pcalda_data(training_tuple, testing_tuple, title=main_option, model_classes=self.pcalda_ctrl.le_classes)
            else:
                self.plotter.ext_plot_pcalda_data(training_tuple, testing_tuple, title=main_option, model_classes=self.pcalda_ctrl.le_classes)'''
        elif main_option in ['PCA Loadings', 'PCA-LDA Loadings']:
            axis = int(''.join(filter(str.isdigit, cur_options['yaxis_option'])))-1
            loadings_tuple = self.pcalda_ctrl.get_loadings((main_option == 'PCA-LDA Loadings'), axis)
            #print('LOADING TUPLE', loadings_tuple[0].shape, loadings_tuple[1].shape)
            if not external:
                self.plotter.clear_data()
                self.plotter.plot_loading_data(loadings_tuple, title=main_option)
            else:
                self.plotter.ext_plot_loading_data(loadings_tuple, title=main_option)
            self.set_state(ControlStates.READY)
    else:
        print("Model has not been trained yet...")
        self.set_state(ControlStates.READY)

def redraw_plot(self):
    self.update_model_data()
    self.set_state(ControlStates.PLOTTING)

def reset_plot(self):
    self.plotter.setup_plot()
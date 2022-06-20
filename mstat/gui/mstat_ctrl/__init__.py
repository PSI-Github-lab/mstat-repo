try:
    import os
    from datetime import datetime
    from functools import partial
    from mstat.dependencies.helper_funcs import *
    from PyQt5 import QtCore, QtWidgets
    from mstat.gui.main_gui import MainGUI
    from mstat.gui.stat_model import PCALDACtrl
    from mstat.gui.plotter import DataPlot
    from mstat.gui.fileselector import FileTreeSelectorModel
    from mstat.gui.table_model import DataTableModel
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class MStatCtrl():
    # structure of this class over multiple files https://stackoverflow.com/questions/3842616/organizing-python-classes-in-modules-and-or-packages
    from mstat.gui.mstat_ctrl.data_funcs import update_model_data, update_tt_dicts, update_conversion_list, convert_RAW, explore_meta
    from mstat.gui.mstat_ctrl.gui_funcs import open_training_folder, open_testing_folder, show_about_message, show_model_info, clear_dir_checks, close_controller
    from mstat.gui.mstat_ctrl.model_funcs import train_model, save_model, delete_model, change_bins, update_bins
    from mstat.gui.mstat_ctrl.plot_funcs import update_plot_data, update_plot_options, redraw_plot, extern_plot, reset_plot, change_model_option, connect_plot_options, disconnect_plot_options

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
        training_dir = self.config_hdlr.get_option('MAIN', 'train directory')
        self.trainingdir_model = FileTreeSelectorModel(rootpath=training_dir)
        testing_dir = self.config_hdlr.get_option('MAIN', 'test directory')
        self.testingdir_model = FileTreeSelectorModel(rootpath=testing_dir)
        
        # Model data table model object
        self.model_data_src = []
        self.table_data_headers = ['Folder', 'Class Name', '# Samples', 'Role']
        self.table_data_model = DataTableModel(self, self.table_data_headers, self.model_data_src)
        self.table_data_model.define_col_flags(1, 1)

        # Create the GUI
        self.gui = MainGUI(self, root_path)

        self.config_hdlr.set_option('MAIN', 'last start time', str(datetime.now()))
        if self.config_hdlr.get_option('MAIN', 'windows num bits') == '64' and self.config_hdlr.get_option('MAIN', 'msfilereader installed') == 'False':
            if not self.gui.show_YN_dialog("Have you installed MSFileReader library from Thermo Scientific?", "MStat 64-bit Requirement"):
                self.gui.showInfo("MSFileReader has not been installed. Please install MSFileReader by following <a href='https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.0/faq/data-import-thermo-raw-need-libraries.aspx#downloading-libraries'>this guide</a><br>Data conversion will not work unless you install MSFileReader.")
                quit()
            else:
                self.config_hdlr.set_option('MAIN', 'msfilereader installed', 'True')
        elif self.config_hdlr.get_option('MAIN', 'windows num bits') == '32' and self.config_hdlr.get_option('MAIN', 'msconvert directory') == '':
            if not self.gui.show_YN_dialog("Have you installed MSFileReader library from Thermo Scientific?", "MStat 32-bit Requirement"):
                self.gui.showInfo("MSConvert has not been recognized. Please located msconvert.exe (first install it by following <a href=''>this link</a>)")
                quit()
            else:
                path = '' # get path from file selector dialog here
                self.config_hdlr.set_option('MAIN', 'msconvert directory', path)

        self.gui.setupStatusBar()       # initialize the status bar at the bottom of the window
        self.gui.attachTTDirModels(self.trainingdir_model, self.testingdir_model, training_dir, testing_dir)   # include the directory models for traingin and testing data
        self.gui.setupModelDataTView(self.table_data_model, self.table_data_headers)      # set of the selected model data table view
        self.gui.show()

        # create PCA-LDA model
        self.pcalda_ctrl = PCALDACtrl(self, self.gui)
        

        # set up the plot view
        self.plotter = DataPlot(self.gui)
        self.plotter.setup_plot()
        
        # Connect signals and slots
        QtCore.QMetaObject.connectSlotsByName(self.gui)
        self.gui.main_view.cleartrainingdata_button.clicked.connect(partial(self.clear_dir_checks, self.trainingdir_model))
        self.gui.main_view.cleartestingdata_button.clicked.connect(partial(self.clear_dir_checks, self.testingdir_model))
        self.gui.main_view.changebin_button.clicked.connect(self.change_bins)
        self.gui.main_view.delmodel_button.clicked.connect(self.delete_model)
        self.gui.main_view.savemodel_button.clicked.connect(self.save_model)
        self.gui.main_view.trainmodel_button.clicked.connect(self.train_model) #partial(self.set_state, ControlStates.BUILDING))
        self.gui.main_view.modelinfo_button.clicked.connect(self.show_model_info)
        self.gui.main_view.exploremeta_button.clicked.connect(self.explore_meta)
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
try:
    import os
    from os.path import exists
    import random
    from mstat.dependencies.helper_funcs import *
    from PyQt5 import QtCore, QtWidgets
    from mstat.gui.mstatDataOptionsDialogUI import Ui_Dialog as DataOptionsDialogGUI
    from mstat.dependencies.file_conversion.RAWConversion import ScanSelAlgorithm, ALGO_NAMES, ALGO_DICT
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class DataOptionsCtrl(QtWidgets.QDialog):
    def __init__(self, main_ctrl, model_ctrl, main_gui):
        super().__init__()
        self.main_gui = main_gui
        self.model_ctrl = model_ctrl
        self.main_ctrl = main_ctrl

        # temporary copy of data_options
        self.data_options = self.main_ctrl.data_options

        do_scan_selection, do_differentiation = self.data_options['perform scan selection'], self.data_options['perform differentiation']
        scan_sel_algo_choice, manual_window, manual_threshold, diff_order = self.data_options['scan selection algorithm'], self.data_options['number of scans'], self.data_options['manual threshold'], self.data_options['differentiation order']
        self.init_bin_size, self.init_low_lim, self.init_up_lim = self.model_ctrl.bin_size, self.model_ctrl.low_lim, self.model_ctrl.up_lim

        self.dialog = DataOptionsDialogGUI()
        self.dialog.setupUi(self)
        self.dialog.buttonBox.accepted.connect(self.dialog_accepted)
        self.dialog.buttonBox.rejected.connect(self.dialog_rejected)

        for name in ALGO_NAMES:
            self.dialog.scanselalgo_combo.addItem(name)

        if do_scan_selection == 'True':
            self.dialog.scan_check.setChecked(True)
        if do_differentiation  == 'True':
            self.dialog.diff_check.setChecked(True)
        if self.init_bin_size > .0:
            self.dialog.binsize_spin.setValue(float(self.init_bin_size))
        if self.init_low_lim > .0:
            self.dialog.lowlim_spin.setValue(float(self.init_low_lim))
        if self.init_up_lim > .0:
            self.dialog.uplim_spin.setValue(float(self.init_up_lim))

        self.dialog.scanselalgo_combo.setCurrentIndex(int(scan_sel_algo_choice))
        self.dialog.numscan_spin.setValue(int(manual_window))
        self.dialog.manualthres_spin.setValue(float(manual_threshold))
        self.dialog.difforder_spin.setValue(int(diff_order))

        self.show()

    def dialog_accepted(self):
        try:
            self.data_options['perform scan selection'] = str(self.dialog.scan_check.isChecked())
            self.data_options['perform differentiation'] = str(self.dialog.diff_check.isChecked())
            self.data_options['scan selection algorithm'] = str(self.dialog.scanselalgo_combo.currentIndex())
            self.data_options['number of scans'] = str(self.dialog.numscan_spin.value())
            self.data_options['manual threshold'] = str(self.dialog.manualthres_spin.value())
            self.data_options['differentiation order'] = str(self.dialog.difforder_spin.value())
            self.main_ctrl.data_options = self.data_options

            bin_size = float(self.dialog.binsize_spin.value())
            low_lim = float(self.dialog.lowlim_spin.value())
            up_lim = float(self.dialog.uplim_spin.value())
            if (self.init_bin_size != bin_size) or (self.init_low_lim != low_lim) or (self.init_up_lim != up_lim):
                self.main_ctrl.update_bins(bin_size, low_lim, up_lim)
        except Exception as exc:
            print(f'From {os.path.basename(__file__)}')
            print(exc)
            self.main_gui.showError("Some binning input values were invalid.")


    def dialog_rejected(self):
        self.close()
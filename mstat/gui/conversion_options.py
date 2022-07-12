try:
    import os
    from os.path import exists
    import random
    from mstat.dependencies.helper_funcs import *
    from PyQt5 import QtCore, QtWidgets
    from mstat.gui.mstatConversionOptionsDialogUI import Ui_Dialog as ConversionOptionsDialogGUI
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class ConversionOptionsCtrl(QtWidgets.QDialog):
    def __init__(self, main_ctrl, main_gui):
        super().__init__()
        self.main_gui = main_gui
        self.main_ctrl = main_ctrl

        conversion_options = self.main_ctrl.conversion_options
        

        do_scan_selection, do_differentiation = conversion_options['perform scan selection'], conversion_options['perform differentiation']
        scan_sel_algo_choice, manual_window, diff_order = conversion_options['scan selection algorithm'], conversion_options['number of scans'], conversion_options['differentiation order']

        self.dialog = ConversionOptionsDialogGUI()
        self.dialog.setupUi(self)
        self.dialog.buttonBox.accepted.connect(self.dialog_accepted)
        self.dialog.buttonBox.rejected.connect(self.dialog_rejected)

        if do_scan_selection:
            self.dialog.scan_check.setChecked(True)
        if do_differentiation:
            self.dialog.diff_check.setChecked(True)
        #if up_lim > .0:
        #    self.dialog.uplim_input.setText(str(up_lim))

        self.dialog.scanselalgo_combo.setCurrentIndex(scan_sel_algo_choice)
        self.dialog.numscan_spin.setValue(int(manual_window))
        self.dialog.difforder_spin.setValue(int(diff_order))
        #self.gui.main_view.showlegend_check.checkState()

        self.show()

    def dialog_accepted(self):
        self.close()


    def dialog_rejected(self):
        self.close()
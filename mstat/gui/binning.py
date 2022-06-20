try:
    import os
    from os.path import exists
    import random
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.file_conversion.RAWConversion import raw_to_numpy_array, run_single_batch
    from PyQt5 import QtCore, QtWidgets
    from mstat.gui.mstatBinningDialogUI import Ui_Dialog as BinningDialogGUI
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class BinningCtrl(QtWidgets.QDialog):
    def __init__(self, main_ctrl, main_gui, model_ctrl):
        super().__init__()
        self.main_gui = main_gui
        self.main_ctrl = main_ctrl
        self.model_ctrl = model_ctrl

        bin_size, low_lim, up_lim = self.model_ctrl.bin_size, self.model_ctrl.low_lim, self.model_ctrl.up_lim

        self.dialog = BinningDialogGUI()
        self.dialog.setupUi(self)
        self.dialog.buttonBox.accepted.connect(self.dialog_accepted)
        self.dialog.buttonBox.rejected.connect(self.dialog_rejected)

        if bin_size > .0:
            self.dialog.bin_size_input.setText(str(bin_size))
        if low_lim > .0:
            self.dialog.lowlim_input.setText(str(low_lim))
        if up_lim > .0:
            self.dialog.uplim_input.setText(str(up_lim))

        self.show()

    def dialog_accepted(self):
        try:
            bin_size = float(self.dialog.bin_size_input.text())
            low_lim = float(self.dialog.lowlim_input.text())
            up_lim = float(self.dialog.uplim_input.text())
            self.main_ctrl.update_bins(bin_size, low_lim, up_lim)
        except Exception as exc:
            print(exc)
            self.main_gui.showError("Some binning input values were invalid.")


    def dialog_rejected(self):
        self.close()
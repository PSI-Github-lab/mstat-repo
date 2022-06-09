from helper_funcs import ControlStates


try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog
    from mstatMainUI import Ui_MainWindow as MainUI
    from mstatConversionDialogUI import Ui_Dialog as ConvDialogUI
except ModuleNotFoundError as e:
    print(e)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

# https://stackoverflow.com/questions/54285057/how-to-include-a-column-of-progress-bars-within-a-qtableview

class ConvDialogGUI(QtWidgets.QDialog):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.dialog_view = ConvDialogUI()
        self.dialog_view.setupUi(self)

    def closeEvent(self, event):
        self.ctrl.main_ctrl.set_state(ControlStates.READY)
        self.ctrl.main_ctrl.update_conversion_list()
        self.close()

    def reattachButton(self, func):
        try: self.dialog_view.beginconversion_button.clicked.disconnect() 
        except Exception: pass
        self.dialog_view.beginconversion_button.clicked.connect(func)

class MainGUI(QtWidgets.QMainWindow):
    def __init__(self, ctrl, root_path):
        super().__init__()
        self.ctrl = ctrl
        self.root_path = root_path
        self.main_view = MainUI()
        self.main_view.setupUi(self)

    def closeEvent(self, event) -> None:
        """
        special method name to overwrite QtWidget closeEvent
        """
        self.ctrl.close_controller()
        print(f"Closing all windows from {event}")
        for window in self.ctrl.app.topLevelWidgets():
            window.close()

    def setupModelDataTView(self, model, header_labels) -> None:
        self.main_view.plotteddata_view.setModel(model)
        self.main_view.plotteddata_view.verticalHeader().setVisible(False)
        self.header = self.main_view.plotteddata_view.horizontalHeader()
        self.header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i in range(1, len(header_labels)):
            self.header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

    def attachTTDirModels(self, trainingdir_model, testingdir_model, training_path='.', testing_path='.') -> None:
        self.trainingdir_model = trainingdir_model
        self.testingdir_model = testingdir_model
        self.setupFolderTView(self.main_view.trainingfolder_tview, trainingdir_model, training_path)
        self.setupFolderTView(self.main_view.testingfolder_tview, testingdir_model, testing_path)

    def setupFolderTView(self, tview, model, path) -> None:
        tview.setModel(model)
        #tview.setObjectName('trainingfolder_tview')

        tview.setAnimated(False)
        tview.setIndentation(10)
        tview.setSortingEnabled(False)
        tview.setColumnWidth(0, 300)

        root_index = model.index(path)
        tview.setRootIndex(root_index)

    def setupStatusBar(self) -> None:
        self.statusBar().showMessage("Ready")

        # create a general purpose button
        self.statusbar_button = QtWidgets.QPushButton('button')
        self.statusBar().addPermanentWidget(self.statusbar_button)
        self.statusbar_button.hide()
        
        # create a general purpose progress bar
        self.statusprogress_bar = QtWidgets.QProgressBar()
        self.statusBar().addPermanentWidget(self.statusprogress_bar)
        self.statusprogress_bar.setGeometry(30, 40, 100, 20)
        self.statusprogress_bar.setValue(0)
        self.statusprogress_bar.hide()

    def setStatusBarMessage(self, message) -> None:
        self.statusBar().showMessage(message)

    def setStatusBarButtonText(self, message) -> None:
        self.statusbar_button.setText(message)

    def reattach_status_bar_button(self, func) -> None:
        try: self.statusbar_button.clicked.disconnect() 
        except Exception: print("status button is not connected to anything")
        self.statusbar_button.clicked.connect(func)

    def reset_status_bar(self) -> None:
        self.statusBar().showMessage("Ready")
        self.statusprogress_bar.setValue(0)
        self.statusprogress_bar.hide()
        self.statusbar_button.hide()
        self.statusbar_button.setEnabled(True)

    def set_xaxis_combo(self, labels, index) -> None:
        self.main_view.xaxis_combo.clear()
        for label in labels:
            self.main_view.xaxis_combo.addItem(label)
        self.main_view.xaxis_combo.setCurrentIndex(index)

    def set_yaxis_combo(self, labels, index) -> None:
        self.main_view.yaxis_combo.clear()
        for label in labels:
            self.main_view.yaxis_combo.addItem(label)
        self.main_view.yaxis_combo.setCurrentIndex(index)
    
    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_trainingfolder_tview_clicked(self, index) -> None:
        print(f'training directory clicked: {self.trainingdir_model.filePath(index)}')
        self.report_dir_checks(self.trainingdir_model, verbose=False)
        #self.ctrl.updateModelData([self.trainingdir_model.filePath(key) for key in list(self.trainingdir_model.checks)], role='Train')

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_testingfolder_tview_clicked(self, index) -> None:
        print(f'testing directory clicked: {self.testingdir_model.filePath(index)}')
        self.report_dir_checks(self.testingdir_model, verbose=False)
        #self.ctrl.updateModelData([self.testingdir_model.filePath(key) for key in list(self.testingdir_model.checks)], role='Test')

    def report_dir_checks(self, model, verbose=False) -> None:
        for key in list(model.checks):
            if model.checks[key] == 0:
                model.checks.pop(key)
            elif verbose:
                print(f"{key.data()} is checked")
        self.ctrl.update_model_data()

    def showMessageWindow(self, message : str, window_title : str, short_text : str, icon) -> int:
        msg = QMessageBox()
        msg.setIcon(icon)
        if short_text is not None:
            msg.setText(f"{short_text}\n\n{message}")
        else:
            msg.setText(message)
        #msg.setInformativeText(message)
        msg.setWindowTitle(window_title)
        return msg.exec_()

    def showError(self, message : str, window_title="Error", short_text=None) -> int:
        return self.showMessageWindow(message, window_title, short_text, QMessageBox.Critical)

    def showInfo(self, message : str, window_title="Information", short_text=None) -> int:
        return self.showMessageWindow(message, window_title, short_text, QMessageBox.Information)

    def folder_dialog(self, open_path, dialog_caption='Select a folder') -> str:
        return QFileDialog.getExistingDirectory(
            self,
            directory=open_path,
            caption=dialog_caption
        )

    def show_YN_dialog(self, message : str, window_title : str) -> bool:
        ret = QMessageBox.question(self, window_title, message, QMessageBox.Yes | QMessageBox.No)

        if ret == QMessageBox.Yes:
            return True
        return False
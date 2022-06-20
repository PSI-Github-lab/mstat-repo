try:
    import os
    from os.path import exists
    import random
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.file_conversion.RAWConversion import raw_to_numpy_array, run_single_batch
    from PyQt5 import QtCore, QtWidgets
    from mstat.gui.main_gui import ConvDialogGUI
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

# https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
# https://stackoverflow.com/questions/9957195/updating-gui-elements-in-multithreaded-pyqt
# https://stackoverflow.com/questions/55779515/add-a-progress-bar-to-a-method-in-pyqt5
# http://biodev.extra.cea.fr/docs/proline/doku.php?id=mzdb_documentation
# https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.0/faq/data-import-thermo-raw-need-libraries.aspx#downloading-libraries

class ConversionWorkerSignals(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int, int, Exception)
    maximumChanged = QtCore.pyqtSignal(int, int, Exception)
    progressComplete = QtCore.pyqtSignal(int, int, Exception)

class ConversionWorker(QtCore.QRunnable):
    def __init__(self, path, row_num, num_files, mzml_check_state):
        super(ConversionWorker, self).__init__()
        self.signals = ConversionWorkerSignals()
        self.path = path
        self.mzml_check = (mzml_check_state == 2)
        self.row_num = row_num
        self.num_files = num_files

    def run(self):
        """
        Convert RAW files to a NPY in a given directory
        """
        try:
            length = self.num_files
            self.signals.maximumChanged.emit(length, self.row_num, Exception())
            i = 0
            mzs = []
            intens = []
            metadata = []
            for file in os.listdir(self.path):
                if file.endswith('raw'):
                    if self.mzml_check:
                        if not exists(rf'{self.path}\MZML'):
                            os.mkdir(rf'{self.path}\MZML')
                        if not exists(rf'{self.path}\MZML\{file.split(".")[0]}.mzml'): # if the file is not already converted, then do it
                            run_single_batch(self.path, file)
                    a, b, c = raw_to_numpy_array(rf'{self.path}\{file}', sel_region=True, smoothing=False)
                    mzs.append(a)
                    intens.append(b)
                    metadata.append(c)
                    i += 1
                    self.signals.progressChanged.emit(i, self.row_num, Exception())

            mzs = np.array(mzs)
            intens = np.array(intens)
            metadata = np.array(metadata)

            file_name = npy_file_name(self.path)
            with open(file_name, 'wb') as f:
                np.save(f, np.array(intens))
                np.save(f, np.array(mzs))
                np.save(f, np.array(metadata))

            print(f"conversion finish items {(i, self.num_files)}")
            if i == self.num_files:
                self.signals.progressComplete.emit(1, self.row_num, Exception())
            else:
                self.signals.progressComplete.emit(0, self.row_num, Exception())
        except Exception as e:
            self.signals.progressComplete.emit(2, self.row_num, e)

class ConversionCtrl:
    def __init__(self, main_ctrl, main_gui, conv_list, model_data):
        self.main_gui = main_gui
        self.main_ctrl = main_ctrl
        self.conversion_list = conv_list
        self.model_data = model_data

        self.progress_total = .0
        self.worker_progress_list = []
        self.num_processes = 0
        self.completed_processes = 0
        
        self.dialog = ConvDialogGUI(self)
        self.dialog.dialog_view.beginconversion_button.clicked.connect(self.startConvWorkers)

        table = self.dialog.dialog_view.convdata_table
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Path', 'Progress', 'Status'])
        
        header = table.horizontalHeader()
        for path in self.conversion_list:
            row_posn = table.rowCount()
            table.insertRow(row_posn)
        
            path_item = QtWidgets.QTableWidgetItem(str(path.absolute()))
            path_item.setToolTip(str(path.absolute()))
            table.setItem(row_posn, 0, path_item)
            table.setCellWidget(row_posn , 1, QtWidgets.QProgressBar())
            table.cellWidget(row_posn, 1).setValue(0)
            table.setItem(row_posn, 2, QtWidgets.QTableWidgetItem("Incomplete"))

        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        self.dialog.show()

        #@QtCore.pyqtSlot(int)
    def on_progressComplete(self, result, row_num, e):
        print(f"\tEnding conversion {self.conversion_list[row_num]}...")
        if result == 1:
            print(f"Complete {row_num}")
            item = QtWidgets.QTableWidgetItem("Completed")
            self.dialog.dialog_view.convdata_table.setItem(row_num, 2, item)
            self.num_processes -= 1
        elif result == 0:
            print(f"Incomplete {row_num}")
            self.num_processes -= 1
            self.main_gui.showError(f"Conversion in \n{self.conversion_list[row_num]}\nfailed.")
            item = QtWidgets.QTableWidgetItem("Failed")
            self.dialog.dialog_view.convdata_table.setItem(row_num, 2, item)
        else:
            print(f"Unknown completion state {row_num}")
            self.num_processes -= 1
            print(e)
            self.main_gui.showError(f"Unknown error occured for \n{self.conversion_list[row_num]}")
            item = QtWidgets.QTableWidgetItem("Failed")
            self.dialog.dialog_view.convdata_table.setItem(row_num, 2, item)

        if self.num_processes == 0:
            print("All processes completed")
            self.progress_total = 0
            self.dialog.dialog_view.beginconversion_button.setText("Finish Conversion")
            self.dialog.dialog_view.beginconversion_button.setEnabled(True)
            self.dialog.reattachButton(self.dialog.close)

            self.main_gui.reset_status_bar()
            output = self.main_gui.showInfo("Conversion completed!")
            self.main_ctrl.set_state(ControlStates.READY)

    def on_progressUpdate(self, result, row_num):
        self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setValue(result)
        self.worker_progress_list[row_num] = result
        #print(f"\t{result} {row_num}")
        self.main_gui.statusprogress_bar.setValue(sum(self.worker_progress_list))

    def on_maximumChanged(self, result, row_num):
        print(f"Maximum {result} {row_num}")
        self.dialog.dialog_view.convdata_table.cellWidget(row_num, 1).setMaximum(result)
        #self.gui.statusprogress_bar.setMaximum(result)

    def startConvWorkers(self):
        #print(self.modeldata_model)
        self.worker_progress_list = []
        for model_num, model_row in enumerate(self.model_data):
            path, num_files = model_row[0], model_row[2]
            if model_row[0] in self.conversion_list:# and model_row[3] == 'Train':
                self.start_worker(path, num_files, self.conversion_list.index(path))
            #elif model_row[0] in self.conversion_list and model_row[3] == 'Test':
            #    self.start_worker(model_row[0], num_files, model_num)
        #for row_num, path in enumerate(self.conversion_list):
        #    num_files = self.model_data[row_num][2]
        #    self.start_worker(path, num_files, row_num)

        self.main_gui.statusprogress_bar.setMaximum(self.progress_total)
        self.dialog.dialog_view.beginconversion_button.setEnabled(False)

    def start_worker(self, path, num_files, row_num):
        cworker = ConversionWorker(str(path.absolute()), row_num, num_files, self.dialog.dialog_view.createmzml_check.checkState())
        cworker.setAutoDelete(True)
        cworker.signals.maximumChanged.connect(self.on_maximumChanged)
        cworker.signals.progressChanged.connect(self.on_progressUpdate)
        cworker.signals.progressComplete.connect(self.on_progressComplete)
        print(f"\tStarting conversion {str(path.absolute())}...")
        self.main_ctrl.threadpool.start(cworker)
        self.worker_progress_list.append(0)
        self.progress_total += num_files
        self.num_processes += 1

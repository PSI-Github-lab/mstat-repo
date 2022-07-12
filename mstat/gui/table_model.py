try:
    from datetime import datetime
    from PyQt5 import QtCore, QtWidgets
    import numpy as np
    import pathlib
    import os
    from pathlib import Path
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

# https://stackoverflow.com/questions/48057638/how-should-i-connect-checkbox-clicked-signals-in-table-widgets-in-pyqt5

class DataTableModel(QtCore.QAbstractTableModel):
    # https://www.pythonguis.com/faq/editing-pyqt-tableview/
    def __init__(self, main_ctrl, header_data, data):
        super(DataTableModel, self).__init__()
        self.main_ctrl = main_ctrl
        self.update_data(data)
        self.header = header_data

    def update_data(self, data_in):
        #print('Updating Table Model')
        # catch if there is no data
        if not data_in:
            data_in = [
                ["","","",""],
            ]
        self._data = data_in
        self._flags = np.zeros((len(self._data), len(self._data[0])))

    def get_data(self):
        return self._data

    def define_col_flags(self, col, val):
        self._flags[:, col] = val * np.ones((self._flags.shape[0],))

    def setData(self, index, value, role):  # special function from super class
        if role == QtCore.Qt.EditRole:
            # update the table data if a field is editted
            #print("edit", value)
            if value != "":
                self._data[index.row()][index.column()] = value
                #print(self._data)
                self.layoutChanged.emit()
                self.dataChanged.emit(index, index)
                self.main_ctrl.update_model_data()
            return True

    def data(self, index, role):    # special function from super class
        if role == QtCore.Qt.DisplayRole:
            # Get the raw value
            value = self._data[index.row()][index.column()]

            # Perform per-type checks and render accordingly.
            if isinstance(value, datetime):
                # Render time to YYYY-MM-DD.
                return value.strftime("%Y-%m-%d")

            if isinstance(value, float):
                # Render float to 4 dp
                return "%.4f" % value

            if isinstance(value, str):
                # Render strings as is
                return value #'"%s"' %  with quotes

            if isinstance(value, pathlib.WindowsPath):
                # Render paths & include tool tip
                dirpath, dir = os.path.split(str(value.absolute()))
                dirpath, parent_dir = os.path.split(dirpath)
                dirpath, grandparent_dir = os.path.split(dirpath)
                dirpath, greatgrandparent_dir = os.path.split(dirpath)
                return f'{greatgrandparent_dir}/{grandparent_dir}/{parent_dir}/{dir}' #'"%s"' % 

            # Default (anything not captured above: e.g. int)
            return value

        if role == QtCore.Qt.ToolTipRole:
            value = self._data[index.row()][index.column()]
            if isinstance(value, pathlib.WindowsPath):
                # Render paths & include tool tip
                return str(value.absolute()) #'"%s"' % 

    def rowCount(self, index):
        # special function from super class
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # special function from super class
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

    def headerData(self, section, orientation, role):
        # special function from super class
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.header[section]
            if orientation == QtCore.Qt.Vertical:
                return f"List {str(section)}" 

    def flags(self, index):
        # special function from super class
        try:
            val = self._flags[index.row(), index.column()]
            if val == 0:
                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            elif val == 1:
                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
        except Exception as e:
            print(f"Exception occured:\n{e}")
            print(f'From {os.path.basename(__file__)}')
            print(f"Model Data table flags:\n{self._flags}")
            return QtCore.Qt.ItemIsSelectable

class MetaTableModel(QtCore.QAbstractTableModel):
    # https://www.pythonguis.com/faq/editing-pyqt-tableview/
    def __init__(self, main_ctrl, header_data, data):
        super(MetaTableModel, self).__init__()
        self.main_ctrl = main_ctrl
        self.update_data(data)
        self.header = header_data

    def update_data(self, data_in):
        #print('Updating Table Model')
        # catch if there is no data
        if not data_in:
            data_in = [
                ["","","",""],
            ]
        self._data = data_in
        self._flags = np.zeros((len(self._data), len(self._data[0])))

    def get_data(self):
        return self._data

    def define_col_flags(self, col, val):
        self._flags[:, col] = val * np.ones((self._flags.shape[0],))

    def data(self, index, role):    # special function from super class
        if role == QtCore.Qt.DisplayRole:
            # Get the raw value
            value = self._data[index.row()][index.column()]

            # Perform per-type checks and render accordingly.
            if isinstance(value, datetime):
                # Render time to YYYY-MM-DD.
                return value.strftime("%Y-%m-%d")

            if isinstance(value, float):
                # Render float to 4 dp
                return "%.4f" % value

            if isinstance(value, str):
                # Render strings as is
                return value #'"%s"' %  with quotes

            if isinstance(value, pathlib.WindowsPath):
                # Render paths & include tool tip
                dirpath, dir = os.path.split(str(value.absolute()))
                dirpath, parent_dir = os.path.split(dirpath)
                dirpath, grandparent_dir = os.path.split(dirpath)
                dirpath, greatgrandparent_dir = os.path.split(dirpath)
                return f'{greatgrandparent_dir}/{grandparent_dir}/{parent_dir}/{dir}' #'"%s"' % 

            # Default (anything not captured above: e.g. int)
            return value

        if role == QtCore.Qt.ToolTipRole:
            value = self._data[index.row()][index.column()]
            if isinstance(value, pathlib.WindowsPath):
                # Render paths & include tool tip
                return str(value.absolute()) #'"%s"' % 

    def rowCount(self, index):  # special function from super class
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):   # special function from super class
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

    def headerData(self, section, orientation, role):   # special function from super class
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.header[section]
            if orientation == QtCore.Qt.Vertical:
                return f"List {str(section)}" 

    def flags(self, index): # special function from super class
        try:
            val = self._flags[index.row(), index.column()]
            if val == 0:
                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            elif val == 1:
                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
            elif val == 2:
                return QtCore.Qt.ItemIsEnabled
        except Exception as e:
            print(f"Exception occured:\n{e}")
            print(f'From {os.path.basename(__file__)}')
            print(f"Model Data table flags:\n{self._flags}")
            return QtCore.Qt.ItemIsSelectable
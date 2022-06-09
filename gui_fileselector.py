"""Dialog to select files or directories
"""
try:
    import sys
    import os
    from PyQt5 import QtCore, QtGui, QtWidgets
except ModuleNotFoundError as e:
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class FileTreeSelectorModel(QtWidgets.QFileSystemModel):
    """
    https://stackoverflow.com/questions/51338059/qfilesystemmodel-qtreeview-traverse-model-filesystem-tree-prior-to-view-e
    """
    def __init__(self, parent=None, rootpath='/'):
        QtWidgets.QFileSystemModel.__init__(self, parent)
        self.root_path      = rootpath
        self.checks         = {}
        self.nodestack      = []
        self.parent_index   = self.setRootPath(self.root_path)
        self.root_index     = self.index(self.root_path) # not used ?

        self.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot)
        self.directoryLoaded.connect(self._loaded)

    def _loaded(self, path):
        print('_loaded', self.root_path, self.rowCount(self.parent_index))

    def clearData(self):
        self.checks = {}

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.CheckStateRole:
            return QtWidgets.QFileSystemModel.data(self, index, role)
        else:
            if index.column() == 0:
                return self.checkState(index)

    def flags(self, index):
        return QtWidgets.QFileSystemModel.flags(self, index) | QtCore.Qt.ItemIsUserCheckable

    def checkState(self, index):
        if index in self.checks:
            return self.checks[index]
        else:
            return QtCore.Qt.Unchecked

    def setData(self, index, value, role):
        if (role == QtCore.Qt.CheckStateRole and index.column() == 0):
            self.checks[index] = value
            #print('setData(): {}'.format(value))
            return True
        return QtWidgets.QFileSystemModel.setData(self, index, value, role)

    def traverseDirectory(self, parentindex, callback=None):
        print('traverseDirectory():')
        callback(parentindex)
        if self.hasChildren(parentindex):
            path = self.filePath(parentindex)
            it = QtCore.QDirIterator(path, self.filter()  | QtCore.QDir.NoDotAndDotDot)
            while it.hasNext():
                childIndex =  self.index(it.next())
                self.traverseDirectory(childIndex, callback=callback)
        else:
            print('no children')

    def printIndex(self, index):
        print(f'model printIndex(): {self.filePath(index)}')


class FileTreeSelectorDialog(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.root_path      = '/Users/caleb/dev/ML/cloudburst-ml/data/test_dir/'

        # Widget
        self.title          = "Application Window"
        self.left           = 200
        self.top            = 100
        self.width          = 1080
        self.height         = 640

        self.setWindowTitle(self.title)         #TODO:  Whilch title?
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Model
        self.model          = FileTreeSelectorModel(rootpath=self.root_path)
        # self.model          = QtWidgets.QFileSystemModel()

        # View
        self.view           = QtWidgets.QTreeView()

        # Attach Model to View
        self.view.setModel(self.model)
        self.view.setRootIndex(self.model.parent_index)

        self.view.setObjectName('treeView_fileTreeSelector')
        self.view.setWindowTitle("Dir View")    #TODO:  Which title?
        self.view.setAnimated(False)
        self.view.setIndentation(10)
        self.view.setSortingEnabled(True)
        self.view.setColumnWidth(0,400)
        self.view.resize(1080, 640)

        # Misc
        self.node_stack     = []

        # GUI
        windowlayout = QtWidgets.QVBoxLayout()
        windowlayout.addWidget(self.view)
        self.setLayout(windowlayout)

        QtCore.QMetaObject.connectSlotsByName(self)

        self.show()

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_treeView_fileTreeSelector_clicked(self, index):
        print(f'tree clicked: {self.model.filePath(index)}')
        #self.model.traverseDirectory(index, callback=self.model.printIndex)
        #print([e.data() for e in list(self.model.checks.keys())])
        #if self.model.checks[index] > 0:
        #    print("Item checked!")
        #else:
        #    print("Item unchecked...")

        for key in list(self.model.checks):
            if self.model.checks[key] == 0:
                self.model.checks.pop(key)
            else:
                print(f"{key.data()} is checked")

        if len(list(self.model.checks)) >= 3:
            #self.model.setRootPath(self.root_path)
            print(list(self.model.checks)[0])
            
            self.model.checks = {}

            

        #for key in self.model.checks:
        #    print(f"{key.data()} is {'checked' if self.model.checks[key] > 0 else 'unchecked'}" )



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = FileTreeSelectorDialog()
    sys.exit(app.exec_())
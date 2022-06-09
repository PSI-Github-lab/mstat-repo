# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mstat_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1203, 769)
        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.horizontal_layout.setSpacing(10)
        self.horizontal_layout.setObjectName("horizontal_layout")
        self.filefolder_layout = QtWidgets.QVBoxLayout()
        self.filefolder_layout.setObjectName("filefolder_layout")
        self.trainingdata_layout = QtWidgets.QHBoxLayout()
        self.trainingdata_layout.setObjectName("trainingdata_layout")
        self.trainingdata_label = QtWidgets.QLabel(self.central_widget)
        self.trainingdata_label.setObjectName("trainingdata_label")
        self.trainingdata_layout.addWidget(self.trainingdata_label)
        self.cleartrainingdata_button = QtWidgets.QPushButton(self.central_widget)
        self.cleartrainingdata_button.setObjectName("cleartrainingdata_button")
        self.trainingdata_layout.addWidget(self.cleartrainingdata_button)
        self.filefolder_layout.addLayout(self.trainingdata_layout)
        self.trainingfolder_tview = QtWidgets.QTreeView(self.central_widget)
        self.trainingfolder_tview.setObjectName("trainingfolder_tview")
        self.filefolder_layout.addWidget(self.trainingfolder_tview)
        self.line_2 = QtWidgets.QFrame(self.central_widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.filefolder_layout.addWidget(self.line_2)
        self.testingdata_layout = QtWidgets.QHBoxLayout()
        self.testingdata_layout.setObjectName("testingdata_layout")
        self.testingdata_label = QtWidgets.QLabel(self.central_widget)
        self.testingdata_label.setObjectName("testingdata_label")
        self.testingdata_layout.addWidget(self.testingdata_label)
        self.cleartestingdata_button = QtWidgets.QPushButton(self.central_widget)
        self.cleartestingdata_button.setObjectName("cleartestingdata_button")
        self.testingdata_layout.addWidget(self.cleartestingdata_button)
        self.filefolder_layout.addLayout(self.testingdata_layout)
        self.testingfolder_tview = QtWidgets.QTreeView(self.central_widget)
        self.testingfolder_tview.setObjectName("testingfolder_tview")
        self.filefolder_layout.addWidget(self.testingfolder_tview)
        self.horizontal_layout.addLayout(self.filefolder_layout)
        self.line_4 = QtWidgets.QFrame(self.central_widget)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontal_layout.addWidget(self.line_4)
        self.model_layout = QtWidgets.QVBoxLayout()
        self.model_layout.setSpacing(10)
        self.model_layout.setObjectName("model_layout")
        self.pcalda_label = QtWidgets.QLabel(self.central_widget)
        self.pcalda_label.setObjectName("pcalda_label")
        self.model_layout.addWidget(self.pcalda_label)
        self.modeloptions_layout = QtWidgets.QGridLayout()
        self.modeloptions_layout.setObjectName("modeloptions_layout")
        self.pcadim_label = QtWidgets.QLabel(self.central_widget)
        self.pcadim_label.setObjectName("pcadim_label")
        self.modeloptions_layout.addWidget(self.pcadim_label, 2, 0, 1, 1)
        self.ldadim_label = QtWidgets.QLabel(self.central_widget)
        self.ldadim_label.setObjectName("ldadim_label")
        self.modeloptions_layout.addWidget(self.ldadim_label, 3, 0, 1, 1)
        self.binsize_edit = QtWidgets.QLineEdit(self.central_widget)
        self.binsize_edit.setObjectName("binsize_edit")
        self.modeloptions_layout.addWidget(self.binsize_edit, 1, 1, 1, 1)
        self.binsize_label = QtWidgets.QLabel(self.central_widget)
        self.binsize_label.setObjectName("binsize_label")
        self.modeloptions_layout.addWidget(self.binsize_label, 1, 0, 1, 1)
        self.ldadim_edit = QtWidgets.QLineEdit(self.central_widget)
        self.ldadim_edit.setEnabled(False)
        self.ldadim_edit.setObjectName("ldadim_edit")
        self.modeloptions_layout.addWidget(self.ldadim_edit, 3, 1, 1, 1)
        self.pcadim_edit = QtWidgets.QLineEdit(self.central_widget)
        self.pcadim_edit.setObjectName("pcadim_edit")
        self.modeloptions_layout.addWidget(self.pcadim_edit, 2, 1, 1, 1)
        self.loadmodel_button = QtWidgets.QPushButton(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadmodel_button.sizePolicy().hasHeightForWidth())
        self.loadmodel_button.setSizePolicy(sizePolicy)
        self.loadmodel_button.setObjectName("loadmodel_button")
        self.modeloptions_layout.addWidget(self.loadmodel_button, 3, 3, 1, 1)
        self.savemodel_button = QtWidgets.QPushButton(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.savemodel_button.sizePolicy().hasHeightForWidth())
        self.savemodel_button.setSizePolicy(sizePolicy)
        self.savemodel_button.setObjectName("savemodel_button")
        self.modeloptions_layout.addWidget(self.savemodel_button, 3, 2, 1, 1)
        self.modelinfo_button = QtWidgets.QPushButton(self.central_widget)
        self.modelinfo_button.setObjectName("modelinfo_button")
        self.modeloptions_layout.addWidget(self.modelinfo_button, 3, 4, 1, 1)
        self.trainmodel_button = QtWidgets.QPushButton(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainmodel_button.sizePolicy().hasHeightForWidth())
        self.trainmodel_button.setSizePolicy(sizePolicy)
        self.trainmodel_button.setObjectName("trainmodel_button")
        self.modeloptions_layout.addWidget(self.trainmodel_button, 1, 2, 2, 3)
        self.model_layout.addLayout(self.modeloptions_layout)
        self.line = QtWidgets.QFrame(self.central_widget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.model_layout.addWidget(self.line)
        self.plotteddata_label = QtWidgets.QLabel(self.central_widget)
        self.plotteddata_label.setObjectName("plotteddata_label")
        self.model_layout.addWidget(self.plotteddata_label)
        self.plotteddata_view = QtWidgets.QTableView(self.central_widget)
        self.plotteddata_view.setObjectName("plotteddata_view")
        self.model_layout.addWidget(self.plotteddata_view)
        self.horizontal_layout.addLayout(self.model_layout)
        self.line_3 = QtWidgets.QFrame(self.central_widget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontal_layout.addWidget(self.line_3)
        self.graph_layout = QtWidgets.QVBoxLayout()
        self.graph_layout.setObjectName("graph_layout")
        self.canvas_layout = QtWidgets.QVBoxLayout()
        self.canvas_layout.setObjectName("canvas_layout")
        self.graph_layout.addLayout(self.canvas_layout)
        self.navigation_layout = QtWidgets.QHBoxLayout()
        self.navigation_layout.setObjectName("navigation_layout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.navigation_layout.addItem(spacerItem)
        self.graph_layout.addLayout(self.navigation_layout)
        self.graphoption_layout = QtWidgets.QGridLayout()
        self.graphoption_layout.setObjectName("graphoption_layout")
        self.yaxis_combo = QtWidgets.QComboBox(self.central_widget)
        self.yaxis_combo.setObjectName("yaxis_combo")
        self.yaxis_combo.addItem("")
        self.graphoption_layout.addWidget(self.yaxis_combo, 3, 1, 1, 1)
        self.xaxis_combo = QtWidgets.QComboBox(self.central_widget)
        self.xaxis_combo.setObjectName("xaxis_combo")
        self.xaxis_combo.addItem("")
        self.graphoption_layout.addWidget(self.xaxis_combo, 3, 0, 1, 1)
        self.extplot_button = QtWidgets.QPushButton(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.extplot_button.sizePolicy().hasHeightForWidth())
        self.extplot_button.setSizePolicy(sizePolicy)
        self.extplot_button.setObjectName("extplot_button")
        self.graphoption_layout.addWidget(self.extplot_button, 1, 4, 3, 1)
        self.xaxis_label = QtWidgets.QLabel(self.central_widget)
        self.xaxis_label.setObjectName("xaxis_label")
        self.graphoption_layout.addWidget(self.xaxis_label, 2, 0, 1, 1)
        self.yaxis_label = QtWidgets.QLabel(self.central_widget)
        self.yaxis_label.setObjectName("yaxis_label")
        self.graphoption_layout.addWidget(self.yaxis_label, 2, 1, 1, 1)
        self.testdata_check = QtWidgets.QCheckBox(self.central_widget)
        self.testdata_check.setObjectName("testdata_check")
        self.graphoption_layout.addWidget(self.testdata_check, 3, 2, 1, 1)
        self.model_combo = QtWidgets.QComboBox(self.central_widget)
        self.model_combo.setObjectName("model_combo")
        self.model_combo.addItem("")
        self.model_combo.addItem("")
        self.model_combo.addItem("")
        self.model_combo.addItem("")
        self.graphoption_layout.addWidget(self.model_combo, 1, 0, 1, 2)
        self.modelplot_label = QtWidgets.QLabel(self.central_widget)
        self.modelplot_label.setObjectName("modelplot_label")
        self.graphoption_layout.addWidget(self.modelplot_label, 0, 0, 1, 3)
        self.showlegend_check = QtWidgets.QCheckBox(self.central_widget)
        self.showlegend_check.setObjectName("showlegend_check")
        self.graphoption_layout.addWidget(self.showlegend_check, 1, 2, 1, 1)
        self.sampleorder_check = QtWidgets.QCheckBox(self.central_widget)
        self.sampleorder_check.setObjectName("sampleorder_check")
        self.graphoption_layout.addWidget(self.sampleorder_check, 2, 2, 1, 1)
        self.graph_layout.addLayout(self.graphoption_layout)
        self.graph_layout.setStretch(0, 1)
        self.horizontal_layout.addLayout(self.graph_layout)
        self.horizontal_layout.setStretch(0, 3)
        self.horizontal_layout.setStretch(2, 1)
        self.horizontal_layout.setStretch(4, 5)
        self.gridLayout.addLayout(self.horizontal_layout, 0, 0, 1, 1)
        self.line_5 = QtWidgets.QFrame(self.central_widget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout.addWidget(self.line_5, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.central_widget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1203, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.statusBar.sizePolicy().hasHeightForWidth())
        self.statusBar.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.statusBar.setFont(font)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionOpen_Training_Folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_Training_Folder.setObjectName("actionOpen_Training_Folder")
        self.actionSave_Current_Analysis = QtWidgets.QAction(MainWindow)
        self.actionSave_Current_Analysis.setObjectName("actionSave_Current_Analysis")
        self.actionOpen_Testing_Folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_Testing_Folder.setObjectName("actionOpen_Testing_Folder")
        self.menuFile.addAction(self.actionAbout)
        self.menuFile.addAction(self.actionOpen_Training_Folder)
        self.menuFile.addAction(self.actionOpen_Testing_Folder)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MStat Main Window"))
        self.trainingdata_label.setText(_translate("MainWindow", "Selected Training Data"))
        self.cleartrainingdata_button.setText(_translate("MainWindow", "Clear Selection"))
        self.testingdata_label.setText(_translate("MainWindow", "Selected Testing Data"))
        self.cleartestingdata_button.setText(_translate("MainWindow", "Clear Selection"))
        self.pcalda_label.setText(_translate("MainWindow", "PCA-LDA Model"))
        self.pcadim_label.setText(_translate("MainWindow", "# of PCs for LDA"))
        self.ldadim_label.setText(_translate("MainWindow", "# of LDs in  LDA"))
        self.binsize_edit.setText(_translate("MainWindow", "1.0"))
        self.binsize_label.setText(_translate("MainWindow", "Bin Size"))
        self.loadmodel_button.setText(_translate("MainWindow", "Load Model"))
        self.savemodel_button.setText(_translate("MainWindow", "Save Model"))
        self.modelinfo_button.setText(_translate("MainWindow", "Model Info"))
        self.trainmodel_button.setText(_translate("MainWindow", "Train Model"))
        self.plotteddata_label.setText(_translate("MainWindow", " Model Data"))
        self.yaxis_combo.setItemText(0, _translate("MainWindow", "PC2"))
        self.xaxis_combo.setItemText(0, _translate("MainWindow", "PC1"))
        self.extplot_button.setText(_translate("MainWindow", "Interactive Plot in New Window"))
        self.xaxis_label.setText(_translate("MainWindow", "x-axis"))
        self.yaxis_label.setText(_translate("MainWindow", "y-axis"))
        self.testdata_check.setText(_translate("MainWindow", "Show Test Data"))
        self.model_combo.setItemText(0, _translate("MainWindow", "PCA Scores"))
        self.model_combo.setItemText(1, _translate("MainWindow", "PCA-LDA Scores"))
        self.model_combo.setItemText(2, _translate("MainWindow", "PCA Loadings"))
        self.model_combo.setItemText(3, _translate("MainWindow", "PCA-LDA Loadings"))
        self.modelplot_label.setText(_translate("MainWindow", "Plot Options"))
        self.showlegend_check.setText(_translate("MainWindow", "Show Legend"))
        self.sampleorder_check.setText(_translate("MainWindow", "Add Sample Order"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionAbout.setText(_translate("MainWindow", "About..."))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionOpen_Training_Folder.setText(_translate("MainWindow", "Open Training Folder"))
        self.actionSave_Current_Analysis.setText(_translate("MainWindow", "Save Current Analysis"))
        self.actionOpen_Testing_Folder.setText(_translate("MainWindow", "Open Testing Folder"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
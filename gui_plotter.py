try:
    import numpy as np
    import sip
    import random
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Circle
    import seaborn as sns
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from helper_funcs import sort_tuple_list
except ModuleNotFoundError as e:
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

CB_color_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        #fig.tight_layout()

class DataPlot():
    def __init__(self, main_gui, training_data=None, testing_data=None):
        self.main_gui = main_gui
        self.training_data = training_data
        self.testing_data = testing_data

        self.setup_options()

        self.canvas = MatplotlibCanvas(self.main_gui.main_view)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self.main_gui.main_view.central_widget)

        self.cmap_opt = 'Dark2'
        self.train_marker = 'o'
        self.test_marker = 'x'

        # various plot variables
        self.train_scatter = None
        self.test_scatter = None

        # connect user interaction events to trigger annotation updates
        #self.canvas.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        #self.canvas.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)

    def setup_plot(self):
        self.clear_data()

    def plot_dummy_data(self):
        ux = [random.random() for _ in range(250)]
        uy = [random.random() for _ in range(250)]
        nx = []
        ny = []
        for x, y in zip(ux, uy):
            nx.append(np.sqrt(-np.log(x)) * np.cos(2*np.pi*y))
            ny.append(np.sqrt(-np.log(x)) * np.sin(2*np.pi*y))

        self.plot_data((nx, ny), 'Normal Scatter')

    def plot_pcalda_data(self, train, test=None, title="PCA Scores"):
        train_data, train_labels, train_pairs = train
        self.plot_scores_data(train_data, train_labels, 'train')

        if test is not None:
            test_data, test_labels, test_pairs = test
            self.plot_scores_data(test_data, test_labels, 'test')

            label_pairs = sort_tuple_list(list(train_pairs.union(test_pairs)), 1)
        else:
            label_pairs = sort_tuple_list(list(train_pairs), 1)
        print('final label pairs', label_pairs)

        labels, colors = list(zip(*label_pairs))

        ax = self.canvas.axes
        if self.options['legend_checked']:
            custom_legend_entries = [Circle((0, 0), color=CB_color_cycle[i], lw=4) for i in colors]
            legend = ax.legend(custom_legend_entries, labels, loc='best')
            legend.set_draggable(True)
        ax.set_title(title)
        ax.set_xlabel(self.options['xaxis_option'])
        ax.set_ylabel(self.options['yaxis_option'])
        self.canvas.fig.tight_layout()
        ax.grid()
        self.canvas.draw()

    def plot_scores_data(self, data, labels, role):
        # data structure (np array, list)
        # (array) score 1, score 2, color # - (list) label
        x, y, colors = data[:, 0], data[:, 1], data[:, 2].astype('int')

        ax = self.canvas.axes
        if role == 'train':
            #self.class_labels = np.unique(labels)
            #self.my_cmap = cm.get_cmap(self.cmap_opt, 1+len(self.class_labels))
            self.train_scatter = ax.scatter(x, y, label=labels, c=CB_color_cycle[colors], marker=self.train_marker)
        else:
            #if np.any(colors >= len(self.class_labels)):
            #    self.class_labels = np.concatenate((self.class_labels, np.array(['unknown'])))
            self.test_scatter = ax.scatter(x, y, label=labels, c=CB_color_cycle[colors], marker=self.test_marker)

    def legend_without_duplicate_labels(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        ax.legend(handle_list, label_list)
        

    def plot_data(self, data, label):
        x, y = data[0], data[1]

        ax = self.canvas.axes
        ax.scatter(x, y, label=label)
        if self.options['legend_checked']:
            legend = ax.legend()
            legend.set_draggable(True)
        
        self.canvas.draw()

    def clear_data(self):
        try:
            self.main_gui.main_view.navigation_layout.removeWidget(self.canvas_toolbar)
            self.main_gui.main_view.canvas_layout.removeWidget(self.canvas)
            sip.delete(self.canvas_toolbar)
            sip.delete(self.canvas)
            self.canvas_toolbar = None
            self.canvas = None
        except Exception as e:
            print("EXCEPTION clearing plot", e)

        self.canvas = MatplotlibCanvas(self.main_gui.main_view)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self.main_gui.main_view.central_widget)

        self.main_gui.main_view.navigation_layout.addWidget(self.canvas_toolbar)
        self.main_gui.main_view.canvas_layout.addWidget(self.canvas)

        self.canvas.axes.cla()
        ax = self.canvas.axes
        ax.set_xlabel('Select data to plot')
        ax.set_ylabel('')
        ax.set_title('No data selected')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def setup_options(self):
        self.options = {
            'legend_checked' : True,
            'sample_order_checked' : False,
            'show_test_checked' : True,
            'model_option' : self.main_gui.main_view.model_combo.currentText(),
            'xaxis_option' : self.main_gui.main_view.xaxis_combo.currentText(),
            'yaxis_option' : self.main_gui.main_view.yaxis_combo.currentText(),
        }

        self.main_gui.main_view.showlegend_check.setChecked(self.options['legend_checked'])
        self.main_gui.main_view.sampleorder_check.setChecked(self.options['sample_order_checked'])
        self.main_gui.main_view.testdata_check.setChecked(self.options['show_test_checked'])
        

    def change_options(self, options):
        #self.plotDummyData()
        # (show_legend, sample_order, show_testdata, model_option, xaxis_option, yaxis_option)
        self.options = {
            'legend_checked' : options[0],
            'sample_order_checked' : options[1],
            'show_test_checked' : options[2],
            'model_option' : options[3],
            'xaxis_option' : options[4],
            'yaxis_option' : options[5],
        }

    def get_options(self):
        return self.options

    def updateTrainingData(self):
        pass

    def updateTestingData(self):
        pass
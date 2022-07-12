try:
    import numpy as np
    import sip
    import os
    import random
    import matplotlib
    import matplotlib.cbook as cbook
    import matplotlib.image as image
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Circle
    import seaborn as sns
    from PyQt5 import QtCore
    from sklearn.preprocessing import LabelEncoder
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from mstat.dependencies.helper_funcs import sort_tuple_list
except ModuleNotFoundError as e:
    import os
    print(e)
    print(f'From {os.path.basename(__file__)}')
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class MatplotlibCanvas(FigureCanvas):
    """
    Class for plotting data in the main window
    """
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        #fig.tight_layout()

class DataPlot():
    CB_color_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])

    from mstat.gui.plot_ctrl.external_plotting import ext_plot_loading_data, ext_plot_pcalda_data

    def __init__(self, main_gui, training_data=None, testing_data=None):
        self.main_gui = main_gui
        self.training_data = training_data
        self.testing_data = testing_data

        script_dir = os.path.dirname(__file__)
        logo_path = "../../img/POINTLOGOarrowhead_128x128.gif"
        abs_logo_path = os.path.join(script_dir, logo_path)

        #print('IMAGE TESTING'.center(80, '*'))
        with cbook.get_sample_data(abs_logo_path) as file:
            raw_im = image.imread(file)
            self.logo_im = []
            for row in raw_im:
                new_row = []
                for item in row:
                    #print(item)
                    if (item[0]==0) and (item[1]==0) and (item[2]==0) and (item[3]==255):
                        new_row.append(np.array([item[0], item[1], item[2], 0]).astype('uint8'))
                    else:
                        new_row.append(np.array(item).astype('uint8'))
                self.logo_im.append(new_row)
        #print(np.array(self.logo).shape)

        # declare default plotting options
        self.setup_options()

        # for plot in main window
        self.canvas = MatplotlibCanvas(self.main_gui.main_view)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self.main_gui.main_view.central_widget)

        self.train_marker = 'o'
        self.test_marker = 'X'
        self.train_scatter = None       # no need for this variable so far
        self.test_scatter = None        # no need for this variable so far

    def setup_plot(self):
        self.clear_data()

    def plot_pcalda_data(self, train=None, test=None, title="PCA Scores", model_classes = None, ax = None):
        """
        Plot scores data from tuples for training and testing data.
        Tuples contain 
            data: (x, y), labels, known classes, DC, DC, DC
        """
        if ax is None:
            ax = self.canvas.axes

        print('MODEL CLASSES', model_classes)
        try:
            # create new encoder to account for the new data
            plot_encoder = LabelEncoder()

            # combine training & testing data + labels, if they exist
            train_data, train_labels, _, _, _, _ = train
            test_data, test_labels, _, _, _, _ = test
            if train_data is not None and test_data is not None:
                data = np.concatenate((train_data, test_data), axis=0)
                labels = np.concatenate((train_labels, test_labels), axis=0)
                print('SELECTED CLASSES', list(np.unique(labels)))
            elif train_data is not None:
                data = train_data
                labels = train_labels
            elif test_data is not None:
                data = test_data
                labels = test_labels
            else:
                raise ValueError('No data to plot')

            '''ax.imshow(self.logo_im,
                      #aspect='auto',
                      extent=(0, 1, 0, 1),
                      alpha=0.25)'''

            # encode the combined labels
            plot_encoder.fit(labels)

            # plot the data
            self.plot_scores_data(data, labels, plot_encoder, model_classes, self.CB_color_cycle)

            # update plot details
            if self.options['legend_checked']:
                u_labels = np.unique(labels)
                custom_legend_entries = [Circle((0, 0), color=self.CB_color_cycle[i], lw=4) for i in plot_encoder.transform(u_labels)]
                legend = ax.legend(custom_legend_entries, u_labels, loc='best')
                legend.set_draggable(True)
            ax.set_title(title)
            ax.set_xlabel(self.options['xaxis_option'])
            ax.set_ylabel(self.options['yaxis_option'])
            self.canvas.fig.tight_layout()

            '''x_frac, y_frac = 0.5, 0.5
            img_x, img_y = 512, 512
            x_offset = int((self.canvas.fig.bbox.xmax * x_frac - img_x/2))
            y_offset = int((self.canvas.fig.bbox.ymax * y_frac - img_y/2))
            self.canvas.fig.figimage(self.logo_im, xo=x_offset, yo=y_offset, origin='upper', zorder=1, alpha=0.25)'''

            #self.canvas.fig.figimage(self.logo_im, 60, 80, zorder=3, alpha=.5)

            ax.grid()
            self.canvas.draw()
        except Exception as exc:
            print(exc)
            print(f'From {os.path.basename(__file__)}')
            print('Could not plot data')

    def plot_scores_data(self, data, labels : list, encoder : LabelEncoder, model_classes : list, color_array : list, ax = None):
        """
        data structure (np array, list, label encoder)
        """
        if ax is None:
            ax = self.canvas.axes
        
        if data is not None:
            x, y = data[:, 0], data[:, 1]
            labels = np.array(labels)

            # create a known vs unknown mask based on the classes recorded by the model controller
            known_mask = np.zeros((labels.shape), dtype='bool')
            for c in model_classes:
                known_mask = np.logical_or(known_mask, (labels == c))
            unknown_mask = np.logical_not(known_mask)
            
            colors = encoder.transform(labels)
            self.train_scatter = ax.scatter(x[known_mask], y[known_mask], label=labels[known_mask], c=color_array[colors[known_mask]], marker=self.train_marker, edgecolors= "black")

            if sum(unknown_mask) > 0:
                self.test_scatter = ax.scatter(x[unknown_mask], y[unknown_mask], label=labels[unknown_mask], c=color_array[colors[unknown_mask]], marker=self.test_marker, edgecolors= "black")

    def plot_loading_data(self, data, title='Loading Plot', ax = None):
        """
        data structure (np array, np array)
        (array) m/z range (array) loadings
        """
        if ax is None:
            ax = self.canvas.axes

        if data is not None:
            mzs, loadings = data
            self.loading_stem = ax.stem(mzs, loadings)
        
        # label plot with axis names, title, et cetera
        ax.set_title(title)
        ax.set_xlabel('m/z (Daltons)')
        ax.set_ylabel('Magnitude')
        ax.grid()

    def plot_dummy_data(self):
        ux = [random.random() for _ in range(250)]
        uy = [random.random() for _ in range(250)]
        nx = []
        ny = []
        for x, y in zip(ux, uy):
            nx.append(np.sqrt(-np.log(x)) * np.cos(2*np.pi*y))
            ny.append(np.sqrt(-np.log(x)) * np.sin(2*np.pi*y))

        self.plot_data((nx, ny), 'Normal Scatter')

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
            print(f'From {os.path.basename(__file__)}')

        self.canvas = MatplotlibCanvas(self.main_gui.main_view)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self.main_gui.main_view.central_widget)

        self.main_gui.main_view.navigation_layout.addWidget(self.canvas_toolbar)
        self.main_gui.main_view.canvas_layout.addWidget(self.canvas)

        self.canvas.axes.cla()
        ax = self.canvas.axes
        ax.set_xlabel('Create model to plot')
        ax.set_ylabel('')
        ax.set_title('No data model')
        self.canvas.fig.tight_layout()

        x_frac, y_frac = 0.5, 0.5
        img_x, img_y = 128, 128
        x_offset = int((self.canvas.fig.bbox.xmax * x_frac - img_x/2))
        y_offset = int((self.canvas.fig.bbox.ymax * y_frac - img_y/2))
        self.canvas.fig.figimage(self.logo_im, xo=x_offset, yo=y_offset, origin='upper', zorder=1, alpha=0.25)
        
        self.canvas.draw()

    def setup_options(self):
        self.options = {
            'legend_checked' : True,
            'sample_order_checked' : False,
            'show_test_checked' : False,
            'model_option' : self.main_gui.main_view.model_combo.currentText(),
            'xaxis_option' : self.main_gui.main_view.xaxis_combo.currentText(),
            'yaxis_option' : self.main_gui.main_view.yaxis_combo.currentText(),
        }

        self.main_gui.main_view.showlegend_check.setChecked(self.options['legend_checked'])
        self.main_gui.main_view.sampleorder_check.setChecked(self.options['sample_order_checked'])
        self.main_gui.main_view.testlabel_check.setChecked(self.options['show_test_checked'])
        

    def change_options(self, options):
        """
        (show_legend, sample_order, show_testdata, model_option, xaxis_option, yaxis_option)
        """
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
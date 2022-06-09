# coding: utf-8
from os import pipe
from bokeh.models.layouts import Row
from matplotlib.markers import MarkerStyle
import numpy as np
from matplotlib import projections, pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import colors
import sys
from datetime import *
import joblib
from numpy.lib.function_base import average
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import pandas as pd

from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import gridplot, row, column
from bokeh.models import ColumnDataSource, Range1d
#from bokeh.io import curdoc
from bokeh.models import HoverTool, Legend, LegendItem, Label, Ellipse
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from dependencies.ms_data.MSFileReader import MSFileReader
import dependencies.visualize.ellipsoid as ellipsoid
import sys

'''
Ideas: 
https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html
https://stats.stackexchange.com/questions/420383/saved-pca-model-produce-different-result
https://wendynavarrete.com/principal-component-analysis-with-numpy/
https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals


COMMAND: bokeh serve --show InteractivePlot_2D.py --args csv_output/quad.csv csv_output/infotest_Test.csv pcalda_results/models/pcalda_quad.model
'''

def close_session(session_context):
    print('closing')
    sys.exit()

class InteractivePlot:
    point_annot = None
    point_label = None

    @staticmethod
    def callback(attr, old, new):
        print('Selected')

    # initialize interactive plot for showing two datasets (one referred to a 'training, the other as 'test')
    def __init__(self, training_file_reader, test_file_reader, pipeline) -> None:
        # load data from training and test data sources
        self.lda_training_data, lda_test_data, training_labels, encoder = self.loadTrainTestData(training_file_reader, test_file_reader, pipeline)
        print(pipeline)
        self.pca = pipeline['pca']
        self.lda = pipeline['lda']

        qda_classifier = QDA(store_covariance=True)
        qda_classifier.fit(self.lda_training_data, training_labels)

        # format the data into dataframes that are easier to plot and keep track of each point source
        training_frame, test_frame = self.projectDataFrames(training_file_reader, test_file_reader, self.lda_training_data, lda_test_data)

        # define colors
        face_color = '#292939'
        title_color = 'white'
        ax1_color = 'white'
        self.colormap = np.array(['navy', 'turquoise', 'darkorange', 'green', 'red', 'chocolate'])

        curdoc().theme = 'dark_minimal'

        # axis 1: Cluster Plot
        # get labels from training data and add the 'unknown' label for test data
        label_names = np.concatenate((encoder.inverse_transform(np.unique(training_labels)), ['unknown']))

        # get colours frome encoded label data and map to plot colors
        training_frame['color'] = self.colormap[encoder.transform(training_frame['label'])]
        test_frame['color'] = self.colormap[max(training_labels) + 1]

        # combine training and testing data to be plotted together
        self.plot_frame = pd.concat([training_frame, test_frame])
        self.plot_frame_cds = ColumnDataSource(self.plot_frame)
        #pd.set_option('display.max_rows', 100)
        print(self.plot_frame)

        # plot the combined data in a scatter plot with the appropriate colors to distinguish classifications
        #self.initAxis1(ax1_color, self.colormap, label_names, qda_classifier)
        TOOLS = ["tap", 'box_zoom', 'pan', 'wheel_zoom','reset']
        clusters = figure(title='PCA-LDA',
                    plot_height=700,#, plot_width=800,
                    tools=TOOLS,
                    x_range=(-12, 12), y_range=(-12, 12),
                    #toolbar_location=None
                    )
        clusters.sizing_mode = 'scale_height'
        
        cluster_tools = [
                ("file", "@filename"),
                ("(x,y)", "(@ld1, @ld2)"),
                ("label", "@label")
            ]

        self.plot_means(clusters)
        self.plot_lda_cov(self.lda, clusters, 2)
        #self.plot_qda_cov(qda_classifier, clusters, 1)

        self.model_plot = clusters.circle('ld1', 'ld2', 
         color='color', legend_field='label', name='cluster_points',
         source= self.plot_frame_cds, size=6, alpha=0.75)

        clusters.add_tools(HoverTool(tooltips=cluster_tools, names=['cluster_points']))

        clusters.add_layout(clusters.legend[0], 'below')
        self.model_plot.data_source.selected.on_change('indices', self.callback)

        # axis 2: Explained variance
        #self.initAxis2(self.lda, title_color)

        # axis 3: Mass Spectrum
        #self.initAxis3(title_color)
        self.spectrum = figure(title='Selected Mass Spectrum',
                    plot_height=700#, plot_width=800
                    #x_range=(-10, 10), y_range=(-10, 10),
                    #toolbar_location=None
                    )
        self.spectrum.sizing_mode = 'scale_height'
        self.spectrum.width = 800

        bins = training_file_reader.bins

        spec = self.spectrum.line(x=bins,y=bins)
        self.mytext = Label(x=average(bins), y=average(bins), text_color='#EEEEEE', text='click cluster point to see mass spectrum')
        self.spectrum.add_layout(self.mytext)
        self.spec_src = spec.data_source


        # connect user interaction events to trigger annotation updates
        #self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        #self.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)
        #show(row(clusters, spectrum))'''
        curdoc().on_session_destroyed(close_session)
        #curdoc().on_server_unloaded(close_session)
        disp_row = row(clusters, self.spectrum)
        disp_row.sizing_mode = 'scale_height'
        curdoc().add_root(disp_row)

    def callback(self, attr, old, new):
        if len(self.model_plot.data_source.selected.indices) > 0:
            ind = self.model_plot.data_source.selected.indices
            name = self.model_plot.data_source.data['filename'][ind]
            print(name)

            inten = self.data_frame[self.data_frame['filename'] == name[0]].values[0][4:]
            self.spec_src.data['y'] = inten
            self.mytext.text = name[0]
            self.mytext.x = min(self.spec_src.data['x'])
            self.mytext.y = max(inten * 0.8)
        

    @staticmethod
    def plot_ellipse(figure, mean, cov, color, n_std):
        proj = np.eye(cov.shape[0])
        cov = proj[:2,:].dot(cov).dot(np.transpose(proj[:2,:]))
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        #angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at n_std standard deviation

        x=mean[0]
        y=mean[1]
        w=2 * n_std * v[0] ** 0.5
        h=2 * n_std * v[1] ** 0.5
        print(w, h)
        #source = ColumnDataSource(dict(x=mean[0], y=mean[1], w=2 * n_std * v[0] ** 0.5, h=2 * n_std * v[1] ** 0.5))
        glyph = Ellipse(x=x, y=y, width=w, height=h, angle=angle, fill_color=color, fill_alpha=0.15)

        figure.add_glyph(glyph)
        


    def plot_qda_cov(self, qda, figure, n_std):
        for i in range(len(qda.means_)):
            self.plot_ellipse(figure, qda.means_[i], qda.covariance_[i], self.colormap[i], n_std)

    def plot_lda_cov(self, lda, figure, n_std):
        print(lda.scalings_.shape)
        print(np.transpose(lda.scalings_).dot(lda.covariance_.dot(lda.scalings_)))
        proj_means = self.lda.means_.dot(self.lda.scalings_)
        for i in range(len(proj_means)):
            #print(qda.covariance_[i])
            self.plot_ellipse(figure, proj_means[i], np.transpose(lda.scalings_).dot(lda.covariance_.dot(lda.scalings_)), self.colormap[i], n_std)

    def plot_means(self, fig):
        proj_means = self.lda.means_.dot(self.lda.scalings_)
        for mean in proj_means:
            fig.x(x=mean[0], y=mean[1], color='black', size=20, alpha=0.85, line_width=3)

    def initAxis1(self, title_color, colormap, label_names, qda_classifier):
        self.sc = self.ax1.scatter(self.plot_frame['pc1'], self.plot_frame['pc2'], color=self.plot_frame['color'])

        # qda ellipses
        self.plot_qda_cov(qda_classifier, self.ax1, 2)

        # means
        self.plot_means()
        
        # label plot with axis names, title, et cetera
        self.ax1.set_title(f'PCA-LDA', color=title_color)
        self.ax1.set_xlabel(f'PC 1', color=title_color)
        self.ax1.set_ylabel(f'PC 2', color=title_color)
        self.ax1.tick_params(axis='x', colors=title_color)
        self.ax1.tick_params(axis='y', colors=title_color)

        # add grid and legend with classification labels
        self.ax1.grid()
        custom_legend_entries = [Circle((0, 0), color=colormap[i], lw=4) for i in range(len(label_names))]
        self.ax1.legend(custom_legend_entries, label_names, loc='best')

    def initAxis2(self, lda, title_color):
        # plot the explained variance data 
        x = np.linspace(1, len(lda.explained_variance_ratio_),len(lda.explained_variance_ratio_))
        cum_var_exp = np.cumsum(100*lda.explained_variance_ratio_)
        self.ax2.bar(x, 100*lda.explained_variance_ratio_)
        self.ax2.step(x, cum_var_exp, where= 'mid')

        # label plot with axis names, title, et cetera
        self.ax2.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        self.ax2.set_ylim([0,108])
        self.ax2.set_title(f'Variance', color=title_color)
        self.ax2.set_xlabel(f'LDA Component', color=title_color)
        self.ax2.set_ylabel(f'Explained Variance Ratio (%)', color=title_color)
        self.ax2.tick_params(axis='x', colors=title_color)
        self.ax2.tick_params(axis='y', colors=title_color)
        self.ax2.grid()

    def initAxis3(self, title_color):
        # label plot with axis names, title, et cetera
        self.ax3.set_title(f'Selected Mass Spectrum', color=title_color)
        self.ax3.set_xlabel(f'm/z (Daltons)', color=title_color)
        self.ax3.set_ylabel(f'Intensity (A.U.)', color=title_color)
        self.ax3.tick_params(axis='x', colors=title_color)
        self.ax3.tick_params(axis='y', colors=title_color)
        self.ax3.grid()

        # annotations / tooltips
        self.point_annot = self.ax1.annotate("", xy=(0,0), xytext=(-130,25), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.3", fc="w", lw=2), arrowprops=dict(arrowstyle='->', color='black', linewidth=2))
        self.mag_annot = self.ax3.annotate("", xy=(0,0), xytext=(5,5), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.1", fc="w", lw=1), arrowprops=dict(arrowstyle='->', color='black', linewidth=2))
        self.spec_annot = self.ax3.annotate("hover over PCA-LDA point to plot spectrum", xy=(0.1,0.8), xytext=(5,5),textcoords="offset points", bbox=dict(boxstyle="round,pad=0.1", fc="w", lw=1))
        self.point_annot.set_visible(False)

    @staticmethod
    def loadTrainTestData(training_file_reader, test_file_reader, pipeline):
        _, training_labels, encoder = training_file_reader.encodeData()
        print(training_file_reader)

        test_file_reader.encodeData()
        print(test_file_reader)

        model_training_data = pipeline.transform(training_file_reader.getTICNormalization())
        model_test_data = pipeline.transform(test_file_reader.getTICNormalization())

        return model_training_data, model_test_data, training_labels, encoder

    @staticmethod
    def loadSingleData(file_reader, pipeline):
        _, training_labels, encoder = file_reader.encodeData()
        print(file_reader)

        pca = pipeline[1]
        lda = pipeline[2]

        lda_data = pipeline.transform(file_reader.getTICNormalization())

        return lda_data, training_labels, lda, encoder

    def projectDataFrames(self, training_file_reader, test_file_reader, lda_training_data, lda_test_data):
        self.data_frame = pd.concat([training_file_reader.file_frame, test_file_reader.file_frame], sort=False)
        
        training_frame = pd.DataFrame({
            'index' : training_file_reader.file_frame.iloc[:,0],
            'label' : training_file_reader.file_frame.iloc[:,1],
            'filename' : training_file_reader.file_frame.iloc[:,2],
            'ld1' : lda_training_data[:,0],
            'ld2' : lda_training_data[:,1]
        })

        test_frame = pd.DataFrame({
            'index' : test_file_reader.file_frame.iloc[:,0],
            'label' : 'unknown',
            'filename' : test_file_reader.file_frame.iloc[:,2],
            'ld1' : lda_test_data[:,0],
            'ld2' : lda_test_data[:,1]
        })

        return training_frame, test_frame

    def projectDataFrame(self, file_reader, lda_data):
        self.data_frame = file_reader.file_frame
        
        frame = pd.DataFrame({
            'index' : file_reader.file_frame.iloc[:,0],
            'label' : file_reader.file_frame.iloc[:,1],
            'filename' : file_reader.file_frame.iloc[:,2],
            'pc1' : lda_data[:,0],
            'pc2' : lda_data[:,1]
        })

        return frame

    def show(self):
        plt.show()

        plt.autoscale(False)

    # cursor hover
    def hover(self, event):
        # check if event was in PCA-LDA plot (axis 1)
        if event.inaxes == self.ax1:    
            # check/get the LDA points contained in the event
            cont, ind = self.sc.contains(event)
            if cont:
                self.highlightPCALDA(event, ind)
            else:
                self.point_annot.set_visible(False)
        # check if event was in spectrum plot (axis 3)
        elif event.inaxes == self.ax3:
            if len(self.ax3.lines) > 0:
                self.highlightSpectrum(event)
        self.fig.canvas.draw_idle()   

    def highlightSpectrum(self, event):
        # show magnitude annotation and reset line variables
        self.mag_annot.set_visible(True)

        try:
            for i in range(1, len(self.ax3.lines)):
                del self.ax3.lines[i]
        except:
            pass

        # calculate position for annotation and vertical line
        index = np.argmin(np.abs(np.array(self.spec_x)-event.xdata))    # closest index to the x posn of the cursor
        self.mag_annot.xy = (event.xdata, event.ydata)
        self.mag_annot.set_text(str(self.spec_y[index]))
        self.ax3.axvline(event.xdata)

        # highlight selected point on PCA-LDA plot
        if self.point_annot != None and self.point_label != None:
            try:
                for i in range(len(self.ax1.lines)):
                    del self.ax1.lines[i]
            except:
                pass

            self.ax1.autoscale(enable=False, axis='both', tight=None)
            point = self.plot_frame[self.plot_frame['filename'] == self.point_label]
            try:
                self.highlight = self.ax1.plot(float(point['pc1']), float(point['pc2']), marker='o', alpha=0.6, markersize=16, fillstyle='none', color='r')
            except:
                print(f'NOTE: Duplicate points from {self.point_label}')
                point = point.iloc[0,:]
                self.highlight = self.ax1.plot(float(point['pc1']), float(point['pc2']), marker='o', alpha=0.6, markersize=16, fillstyle='none', color='r')


    def highlightPCALDA(self, event, ind):
        # change LDA point annotation position
        self.point_annot.xy = (event.xdata, event.ydata)

        # update label to select a single point
        names = self.plot_frame['filename'].values
        self.point_label = names[int(ind['ind'][0])]

        # reset axis 3
        try:
            for i in range(len(self.ax3.lines)):
                del self.ax3.lines[i]
        except:
            pass

        # get spectral data for selected point
        inten = self.data_frame[self.data_frame['filename'] == self.point_label].values[0][4:]
        norm = inten / max(inten)
        bins = [float(e) for e in list(self.data_frame[self.data_frame['filename'] == self.point_label].columns)[4:]]

        # plot spectral data
        self.spec_x = bins
        self.spec_y = inten
        self.ax3.plot(self.spec_x, self.spec_y, color='black')
        self.ax3.set_ylim([0,1.2*max(self.spec_y)])
        self.spec_annot.xy = (0.85*min(self.spec_x), 1.02*max(self.spec_y))

        # update annotation text
        self.point_annot.set_text(self.point_label)
        self.spec_annot.set_text(self.point_label)
        self.point_annot.set_visible(True)   

    # when leaving any axis clean axis 2 and hide annotation
    def leave_axes(self, event):
        self.point_annot.set_visible(False)
        self.mag_annot.set_visible(False)

        try:
            for i in range(1, len(self.ax3.lines)):
                del self.ax3.lines[i]
        except:
            pass

        try:
            for i in range(len(self.ax1.lines)):
                del self.ax1.lines[i]
        except:
            pass

help_message = """
Console Command: python pcaldaPlot.py <model_data_name.csv> <test_data_name.csv> <pcalda_model.model>
Arguments:
    <path/model_data_name.csv> - (String) first CSV file including the extension ".csv"
    <path/test_data_name.csv>  - (String) second CSV file which will be plotted and labelled with first CSV file
    <path/pcalda_model.model>  - (String) PCA-LDA model file"""

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main(argm):
    ''' CONSOLE INPUT: 
    python pcaldaPlot.py <file_name.extension>
    OR
    python pcaldaPlot.py
    OR
    python pcaldaPlot.py help
    '''
    # handle user commands
    #argm = handleStartUpCommands(help_message)
    if not argm:
        print("Type 'python pcaldaPlot.py help' for more info")
        quit()
    else:
        first_csv_file = argm[0]
        second_csv_file = argm[1]
        pcalda_file = argm[2]

    # read data from the csv files
    training_file_reader = MSFileReader(first_csv_file)
    test_file_reader = MSFileReader(second_csv_file)

    # load saved pcalda model
    pipeline = joblib.load(pcalda_file)
    print(pipeline)

    plot = InteractivePlot(training_file_reader, test_file_reader, pipeline)
    #plot.show()

'''
def modify_doc(doc):
    #main(doc)
    p = figure()
    p.line([1,2,3,4,5], [3,4,2,7,5], line_width=2)
    doc.add_root(p)

if __name__ == "__main__":
    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop())
    server.start()
    server.io_loop.start()'''


print(sys.argv[1:])

main(sys.argv[1:])
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
import sys
from datetime import *
import joblib
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import pandas as pd

from dependencies.ms_data.MSFileReader import MSFileReader
import dependencies.visualize.ellipsoid as ellipsoid

'''
Ideas: 
https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html
https://stats.stackexchange.com/questions/420383/saved-pca-model-produce-different-result
https://wendynavarrete.com/principal-component-analysis-with-numpy/
https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
'''

class InteractivePlot:
    point_annot = None
    point_label = None

    # initialize interactive plot for showing two datasets (one referred to a 'training, the other as 'test')
    def __init__(self, training_file_reader, test_file_reader, pipeline, n_std) -> None:
        # load data from training and test data sources
        self.lda_training_data, lda_test_data, training_labels, test_predictions, encoder = self.loadTrainTestData(training_file_reader, test_file_reader, pipeline)
        self.pca = pipeline['dim']
        self.lda = pipeline['lda']

        qda_classifier = QDA(store_covariance=True)
        qda_classifier.fit(self.lda_training_data, training_labels)

        # format the data into dataframes that are easier to plot and keep track of each point source
        training_frame, test_frame = self.projectDataFrames(training_file_reader, test_file_reader, self.lda_training_data, lda_test_data)

        # define colors
        face_color = '#BBBBBB'
        title_color = 'black'
        ax1_color = 'black'
        self.colormap = cm.get_cmap('rainbow', 1+len(np.unique(training_labels)))

        # create figure
        self.fig = plt.figure(figsize=(8,8), facecolor=face_color)
        gs = gridspec.GridSpec(ncols=4, nrows=2, wspace=0.5, hspace=0.5, figure=self.fig)
        #self.ax2 = self.fig.add_subplot(gs[0, 2:])
        #self.ax3 = self.fig.add_subplot(gs[1, 2:])
        self.ax1 = self.fig.add_subplot(gs[0:4, 0:4], projection='3d')   # want this to be plotted last so annotation rise to the top
        plt.suptitle('PCA-LDA Analysis of MS Data', color=title_color) 

        # axis 1: Cluster Plot
        # get labels from training data and add the 'unknown' label for test data
        label_names = np.unique(training_labels)

        # get colours frome encoded label data and map to plot colors
        training_frame['color'] = encoder.transform(training_frame['label'])
        if test_frame is not None:
            test_frame['color'] = encoder.transform(test_predictions)

            # combine training and testing data to be plotted together
            self.plot_frame = pd.concat([training_frame, test_frame])
        else:
            self.plot_frame = training_frame

        print(' PLOTTED DATA FRAME'.center(80, '*'))
        print(self.plot_frame)

        # plot the combined data in a scatter plot with the appropriate colors to distinguish classifications
        self.initAxis1(ax1_color, training_frame, test_frame, self.colormap, label_names, qda_classifier, n_std)

    def plot_qda_cov_3d(self, qda, ax, n_std):
        for i in range(len(qda.means_)):
            X1,Y1,Z1 = ellipsoid.get_cov_ellipsoid(qda.covariance_[i], qda.means_[i], n_std)
            self.ax1.plot_surface(X1,Y1,Z1, rstride=4, cstride=4, color=self.colormap(i), alpha=0.4)

    def plot_means_3d(self):
        proj_means = self.lda.means_.dot(self.lda.scalings_)
        for mean in proj_means:
            self.ax1.scatter(mean[0], mean[1], mean[2], c='black', marker='x', s=100, linewidths=3)

    def initAxis1(self, title_color, training_frame, test_frame, colormap, label_names, qda_classifier, n_std):
        self.sctr = self.ax1.scatter(training_frame['ld1'], training_frame['ld2'], training_frame['ld3'], color=colormap(training_frame['color']), zorder=-1)
        if test_frame is not None:
            self.scts = self.ax1.scatter(test_frame['ld1'], test_frame['ld2'], test_frame['ld3'], color=colormap(test_frame['color']), marker='X', s=160, edgecolors='black', zorder=99)

        # qda ellipses
        #self.plot_qda_cov_3d(qda_classifier, self.ax1, n_std)

        # means
        #self.plot_means_3d()

        # label plot with axis names, title, et cetera
        self.ax1.set_title('PCA-LDA', color=title_color)
        self.ax1.set_xlabel('LD 1', color=title_color)
        self.ax1.set_ylabel('LD 2', color=title_color)
        self.ax1.set_zlabel('LD 3', color=title_color)
        self.ax1.tick_params(axis='x', colors=title_color)
        self.ax1.tick_params(axis='y', colors=title_color)
        self.ax1.tick_params(axis='z', colors=title_color)

        # add grid and legend with classification labels
        self.ax1.grid()
        custom_legend_entries = [Circle((0, 0), color=colormap(i), lw=4) for i in range(len(label_names))]
        self.ax1.legend(custom_legend_entries, label_names, loc='best')

    @staticmethod
    def loadTrainTestData(training_file_reader, test_file_reader, pipeline):
        _, _, training_labels, encoder = training_file_reader.encodeData()
        print(' DATA FROM CSV FILE '.center(80, '*'))
        print(training_file_reader)
        model_training_data = pipeline.transform(training_file_reader.getTICNormalization())

        if test_file_reader is None:
            print("No test file given. Only ploting training data.")
            model_test_data = None
            test_predictions = None
        else:
            test_file_reader.encodeData()
            print(test_file_reader)
            tdata = test_file_reader.getTICNormalization()
            model_test_data = pipeline.transform(tdata)
            test_predictions = pipeline.predict(tdata)

        return model_training_data, model_test_data, training_labels, test_predictions, encoder

    @staticmethod
    def loadSingleData(file_reader, pipeline):
        _, training_labels, encoder = file_reader.encodeData()
        print(file_reader)

        pca = pipeline[1]
        lda = pipeline[2]

        lda_data = pipeline.transform(file_reader.getTICNormalization())

        return lda_data, training_labels, lda, encoder

    def projectDataFrames(self, training_file_reader, test_file_reader, lda_training_data, lda_test_data):
        if lda_test_data is None or test_file_reader is None:
            self.data_frame = training_file_reader.file_frame
        else:
            self.data_frame = pd.concat([training_file_reader.file_frame, test_file_reader.file_frame], sort=False)
        
        if lda_training_data.shape[1] > 1:
            training_frame = pd.DataFrame({
                'index' : training_file_reader.file_frame.iloc[:,0],
                'label' : training_file_reader.file_frame.iloc[:,1],
                'filename' : training_file_reader.file_frame.iloc[:,2],
                'ld1' : lda_training_data[:,0],
                'ld2' : lda_training_data[:,1],
                'ld3' : lda_training_data[:,2]
            })

            if lda_test_data is None:
                test_frame = None
            else:
                test_frame = pd.DataFrame({
                    'index' : test_file_reader.file_frame.iloc[:,0],
                    'label' : 'unknown',
                    'filename' : test_file_reader.file_frame.iloc[:,2],
                    'ld1' : lda_test_data[:,0],
                    'ld2' : lda_test_data[:,1],
                    'ld3' : lda_test_data[:,2]
                })
        else:
            training_frame = pd.DataFrame({
                'index' : training_file_reader.file_frame.iloc[:,0],
                'label' : training_file_reader.file_frame.iloc[:,1],
                'filename' : training_file_reader.file_frame.iloc[:,2],
                'ld1' : lda_training_data[:,0]
            })

            if lda_test_data is None:
                test_frame = None
            else:
                test_frame = pd.DataFrame({
                    'index' : test_file_reader.file_frame.iloc[:,0],
                    'label' : 'unknown',
                    'filename' : test_file_reader.file_frame.iloc[:,2],
                    'ld1' : lda_test_data[:,0]
                })

        return training_frame, test_frame

    def projectDataFrame(self, file_reader, lda_data):
        self.data_frame = file_reader.file_frame

        return pd.DataFrame({
            'index' : file_reader.file_frame.iloc[:,0],
            'label' : file_reader.file_frame.iloc[:,1],
            'filename' : file_reader.file_frame.iloc[:,2],
            'ld1' : lda_data[:,0],
            'ld2' : lda_data[:,1]
        })

    def show(self):
        plt.show()

        plt.autoscale(False)

    # cursor hover
    def hover(self, event):
        print((event.xdata, event.ydata, event.zdata))
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
                self.highlight = self.ax1.plot(float(point['ld1']), float(point['ld2']), float(point['ld3']), marker='o', alpha=0.6, markersize=16, fillstyle='none', color='r')
            except:
                print(f'NOTE: Duplicate points from {self.point_label}')
                point = point.iloc[0,:]
                self.highlight = self.ax1.plot(float(point['ld1']), float(point['ld2']), float(point['ld3']) , marker='o', alpha=0.6, markersize=16, fillstyle='none', color='r')


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
Console Command: python pcaldaPlot.py <model_data_name.csv> <test_data_name.csv> <pcalda_model.model> <n_std>
Arguments:
    <path/model_data_name.csv> - (String) first CSV file including the extension ".csv"
    <path/test_data_name.csv>  - (String) second CSV file which will be plotted and labelled with first CSV file
    <path/pcalda_model.model>  - (String) PCA-LDA model file
    <n_std>                    - (Float) Number of standard deviations that the confidence ellipsoids should extend to"""

def handleStartUpCommands(help_message):
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
    ''' CONSOLE INPUT: 
    python pcaldaPlot.py <file_name.extension>
    OR
    python pcaldaPlot.py
    OR
    python pcaldaPlot.py help
    '''
    if argm := handleStartUpCommands(help_message):
        first_csv_file = argm[0]

        if len(argm) == 3:
            test_file_reader = None
            pcalda_file = argm[1]
            n_std = float(argm[2])
        else:
            second_csv_file = argm[1]
            pcalda_file = argm[2]
            n_std = float(argm[3])

            test_file_reader = MSFileReader(second_csv_file)
        training_file_reader = MSFileReader(first_csv_file)

    else:
        print("Type 'python pcaldaPlot.py help' for more info")
        quit()

    # load saved pcalda model
    pred_est, _, meta_info = joblib.load(pcalda_file)
    try:
        print(f"\nModel trained with the following data: {meta_info['training_file']}\n")
    except:
        print("ERROR: Improper or corrupted model file. Choose another file that stores a PCA-LDA model.")
        quit()

    try:
        pred_est['lda']
    except KeyError as e:
        #print(e)
        print("ERROR: Must use a PCA-LDA model. Choose another file that stores a PCA-LDA model.")
        quit()
    print(pred_est)

    plot = InteractivePlot(training_file_reader, test_file_reader, pred_est, n_std)
    plot.show()

if __name__ == "__main__":
    main()
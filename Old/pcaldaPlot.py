# coding: utf-8
import numpy as np
from matplotlib import projections, pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import sys
from datetime import *
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

from dependencies.ms_data.MSFileReader import MSFileReader

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

    def __init__(self, training_file_reader, test_file_reader, pca, lda) -> None:
        # load data from training and test data files
        training_data, training_labels, encoder = training_file_reader.encodeData()
        print(training_file_reader)

        test_data, test_labels, _ = test_file_reader.encodeData()
        print(test_file_reader)

        pca_training_data = pca.transform(training_file_reader.getTICNormalization())
        lda_training_data = lda.transform(pca_training_data)

        pca_test_data = pca.transform(test_file_reader.getTICNormalization())
        lda_test_data = lda.transform(pca_test_data)

        # format the data into dataframes that are easier to plot and keep track of each point source
        self.data_frame = pd.concat([training_file_reader.file_frame, test_file_reader.file_frame])
        training_frame = pd.DataFrame({
            'index' : training_file_reader.file_frame.iloc[:,0],
            'label' : training_file_reader.file_frame.iloc[:,1],
            'filename' : training_file_reader.file_frame.iloc[:,2],
            'pc1' : lda_training_data[:,0],
            'pc2' : lda_training_data[:,1]
        })

        test_frame = pd.DataFrame({
            'index' : test_file_reader.file_frame.iloc[:,0],
            'label' : 'unknown',
            'filename' : test_file_reader.file_frame.iloc[:,2],
            'pc1' : lda_test_data[:,0],
            'pc2' : lda_test_data[:,1]
        })

        face_color = '#292939'
        title_color = 'white'
        colormap = np.array(['navy', 'turquoise', 'darkorange', 'red', 'green', 'chocolate'])

        # create figure
        self.fig = plt.figure(figsize=(16,7), facecolor=face_color)
        gs = gridspec.GridSpec(ncols=4, nrows=2, wspace=0.5, hspace=0.5, figure=self.fig)
        self.ax2 = self.fig.add_subplot(gs[0, 2:])
        self.ax3 = self.fig.add_subplot(gs[1, 2:])
        self.ax1 = self.fig.add_subplot(gs[0:2, 0:2])   # want this to be plotted last so annotation rise to the top
        plt.suptitle('PCA-LDA Analysis of MS Data', color=title_color) 

        # axis 1: Cluster Plot
        # get labels from training data and add the 'unknown' label for test data
        labels = training_labels
        label_names = np.concatenate((encoder.inverse_transform(np.unique(labels)), ['unknown']))

        # get colours frome encoded label data and map to plot colors
        training_frame['color'] = colormap[encoder.transform(training_frame['label'])]
        test_frame['color'] = colormap[max(labels) + 1]

        # combine training and testing data to be plottde together
        self.plot_frame = pd.concat([training_frame, test_frame])
        pd.set_option('display.max_rows', 100)
        print(self.plot_frame)

        # plot the combined data in a scatter plot with the appropriate colors to distinguish classifications
        self.sc = self.ax1.scatter(self.plot_frame['pc1'], self.plot_frame['pc2'], color=self.plot_frame['color'])
        
        # label plot with axis names, title, et cetera
        exp1 = float('%.4g' % (100*lda.explained_variance_ratio_[0]))
        exp2 = float('%.4g' % (100*lda.explained_variance_ratio_[1]))
        self.ax1.set_title(f'PCA-LDA', color=title_color)
        self.ax1.set_xlabel(f'PC 1 ({exp1}%)', color=title_color)
        self.ax1.set_ylabel(f'PC 2 ({exp2}%)', color=title_color)
        self.ax1.tick_params(axis='x', colors=title_color)
        self.ax1.tick_params(axis='y', colors=title_color)

        # add grid and legend with classification labels
        self.ax1.grid()
        custom_legend_entries = [Circle((0, 0), color=colormap[i], lw=4) for i in range(len(label_names))]
        self.ax1.legend(custom_legend_entries, label_names, loc='best')

        # axis 2: Explained variance
        # plot the explained variance data 
        x = np.linspace(1, len(lda.explained_variance_ratio_),len(lda.explained_variance_ratio_))
        self.ax2.stem(x, 100*lda.explained_variance_ratio_)

        # label plot with axis names, title, et cetera
        self.ax2.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        self.ax2.set_ylim([0,100])
        self.ax2.set_title(f'Variance', color=title_color)
        self.ax2.set_xlabel(f'LDA Component', color=title_color)
        self.ax2.set_ylabel(f'Explained Variance (%)', color=title_color)
        self.ax2.tick_params(axis='x', colors=title_color)
        self.ax2.tick_params(axis='y', colors=title_color)
        self.ax2.grid()

        # axis 3: Mass Spectrum
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

        # connect user interaction events to trigger annotation updates
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)

    def show(self):
        plt.show()

        plt.autoscale(False)

    # cursor hover
    def hover(self, event):
        # check if event was in axis 1
        if event.inaxes == self.ax1:        
            # get the LDA points contained in the event
            cont, ind = self.sc.contains(event)
            if cont:
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
            
            # when stop hovering a point hide annotation
            else:
                self.point_annot.set_visible(False)
        # check if event was in axis 2

        # check if event was in axis 3
        elif event.inaxes == self.ax3:
            if len(self.ax3.lines) > 0:
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
                    self.highlight = self.ax1.plot(float(point['pc1']), float(point['pc2']), alpha=0.6, markersize=16, marker='o', fillstyle='none', color='red')
        
        self.fig.canvas.draw_idle()   

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

def main():
    ''' CONSOLE INPUT: 
    python pcaldaPlot.py <file_name.extension>
    OR
    python pcaldaPlot.py
    OR
    python pcaldaPlot.py help
    '''
    # handle user commands
    argm = []
    for arg in sys.argv[1:]:
        argm.append(arg)

    if len(argm) == 0:
        print("Input first CSV file name:")
        first_csv_file = input()
        print("Input second CSV file name:")
        second_csv_file = input()
        print("Input PCA-LDA model file name:")
        pcalda_file = input()
    else:
        if(argm[0] == 'help'):
            print("""
Console Command: python pcaldaPlot.py <model_data_name.csv> <test_data_name.csv> <pcalda_model.model>
Arguments:
    <path/model_data_name.csv> - (String) first CSV file including the extension ".csv"
    <path/test_data_name.csv>  - (String) second CSV file which will be plotted and labelled with first CSV file
    <path/pcalda_model.model>  - (String) PCA-LDA model file""")
            quit()
        else:
            first_csv_file = argm[0]
            second_csv_file = argm[1]
            pcalda_file = argm[2]

    # read data from the csv files
    training_file_reader = MSFileReader(first_csv_file)
    test_file_reader = MSFileReader(second_csv_file)

    # load saved pcalda model
    pca, lda = joblib.load(pcalda_file)

    # means
    ##proj_means = lda.means_.dot(lda.scalings_)
    ##for mean in proj_means:
    ##    plt.plot(mean[0], mean[1], '*', color='yellow', markersize=15, markeredgecolor='grey')
             
    plot = InteractivePlot(training_file_reader, test_file_reader, pca, lda)
    plot.show()

if __name__ == "__main__":
    main()
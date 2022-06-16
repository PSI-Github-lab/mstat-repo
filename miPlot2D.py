# coding: utf-8
try:
    from matplotlib.markers import MarkerStyle
    import numpy as np
    from matplotlib import projections, pyplot as plt, cm
    from matplotlib.patches import Circle
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl
    from matplotlib import colors
    import sys
    from datetime import *
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    import pandas as pd
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

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
    num_training = 0

    # initialize interactive plot for showing two datasets (one referred to a 'training, the other as 'test')
    def __init__(self, training_file_reader, test_file_reader, pipeline, n_std) -> None:
        # load data from training and test data sources
        self.lda_training_data, lda_test_data, training_labels, test_predictions, encoder = self.loadTrainTestData(training_file_reader, test_file_reader, pipeline)
        self.pca = pipeline['dim']
        self.lda = pipeline['lda']

        self.qda_classifier = QDA(store_covariance=True)
        self.qda_classifier.fit(self.lda_training_data, training_labels)

        # format the data into dataframes that are easier to plot and keep track of each point source
        training_frame, test_frame = self.projectDataFrames(training_file_reader, test_file_reader, self.lda_training_data, lda_test_data, test_predictions)
        self.num_training = len(training_frame.iloc[:,0])

        # define colors
        face_color = '#292939'
        title_color = 'white'
        ax1_color = 'white'
        self.colormap = cm.get_cmap('rainbow', 1+len(np.unique(training_labels)))

        # create figure
        self.fig = plt.figure(1, figsize=(16,7), facecolor=face_color)
        gs = gridspec.GridSpec(ncols=4, nrows=2, wspace=0.5, hspace=0.5, figure=self.fig)
        self.ax2 = self.fig.add_subplot(gs[0, 2:])
        self.ax3 = self.fig.add_subplot(gs[1, 2:])
        self.ax1 = self.fig.add_subplot(gs[0:2, 0:2])   # want this to be plotted last so annotations rise to the top
        plt.suptitle('PCA-LDA Analysis of MS Data', color=title_color) 

        # axis 1: Cluster Plot
        # get labels from training data and add the 'unknown' label for test data
        self.label_names = np.unique(training_labels)

        # get colours frome encoded label data and map to plot colors
        training_frame['color'] = encoder.transform(training_frame['label'])
        if test_frame is not None:
            test_frame['color'] = encoder.transform(test_predictions)

            # combine training and testing data to be plotted together
            self.plot_frame = pd.concat([training_frame, test_frame])
        else:
            self.plot_frame = training_frame
        #pd.set_option('display.max_rows', 100)
        print(' PLOTTED DATA FRAME '.center(80, '*'))
        print(self.plot_frame)

        # plot the combined data in a scatter plot with the appropriate colors to distinguish classifications
        self.scores_x, self.scores_y, _, _ = self.initAxis1(self.ax1, training_frame, test_frame, ax1_color, n_std)

        # axis 2: Explained variance
        _, self.loadings = self.initAxis2(self.lda, title_color)

        # axis 3: Mass Spectrum
        self.initAxis3(title_color)

        # connect user interaction events to trigger annotation updates
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)

    @staticmethod
    def plot_ellipse(ax, mean, cov, color, n_std):
        proj = np.eye(cov.shape[0])
        cov = proj[:2,:].dot(cov).dot(np.transpose(proj[:2,:]))
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at n standard deviation
        ell = mpl.patches.Ellipse(mean, 2 * n_std * v[0] ** 0.5, 2 * n_std * v[1] ** 0.5,
                                180 + angle, facecolor=color,
                                edgecolor='black', linewidth=0.0)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    def plot_qda_cov(self, qda, ax, n_std):
        for i in range(len(qda.means_)):
            #print(qda.covariance_[i])
            self.plot_ellipse(ax, qda.means_[i], qda.covariance_[i], self.colormap(i), n_std)

    def plot_lda_cov(self, lda, ax, n_std):
        print(lda.scalings_.shape)
        print(np.transpose(lda.scalings_).dot(lda.covariance_.dot(lda.scalings_)))
        proj_means = lda.means_.dot(lda.scalings_)
        for i in range(len(proj_means)):
            #print(qda.covariance_[i]) np.transpose(lda.scalings_).dot(lda.covariance_.dot(lda.scalings_))
            self.plot_ellipse(ax, proj_means[i], np.transpose(lda.scalings_).dot(lda.covariance_.dot(lda.scalings_)), self.colormap(i), n_std)

    def plot_means(self, ax):
        proj_means = self.lda.means_.dot(self.lda.scalings_)
        for mean in proj_means:
            try:
                self.ax1.scatter(mean[0], mean[1], c='black', marker='o', s=100, linewidths=3)
            except:
                self.ax1.scatter(mean[0], 0, c='black', marker='o', s=100, linewidths=3)

    def initAxis1(self, ax, training_frame, test_frame, title_color, n_std):
        label_names = self.label_names
        qda_classifier = self.qda_classifier
        colormap = self.colormap
        xtr = training_frame['ld1']
        xts = test_frame['ld1'] if test_frame is not None else []
        try:
            ytr = training_frame['ld2']
            yts = test_frame['ld2'] if test_frame is not None else []
            dim_flag = True
        except KeyError:
            ytr = np.zeros(xtr.shape)
            yts = np.zeros(xts.shape)
            dim_flag = False

        if dim_flag:
            #self.plot_lda_cov(self.lda, self.ax1, n_std)
            self.plot_qda_cov(qda_classifier, ax, n_std)

        if test_frame is not None:
            self.scts = ax.scatter(xts, yts, color=self.colormap(test_frame['color']), marker='X', s=160, edgecolors='black')
        else:
            self.scts = None

        self.sctr = ax.scatter(xtr, ytr, color=self.colormap(training_frame['color']))

        # means
        #self.plot_means(ax)

        # label plot with axis names, title, et cetera
        ax.set_title('PCA-LDA Score Plot', color=title_color)
        ax.set_xlabel('LD 1', color=title_color)
        ax.set_ylabel('LD 2', color=title_color)
        ax.tick_params(axis='x', colors=title_color)
        ax.tick_params(axis='y', colors=title_color)
        #self.ax1.set_xlim([-12, 12])
        #self.ax1.set_ylim([-12, 12])

        # add grid and legend with classification labels
        self.ax1.grid()
        custom_legend_entries = [Circle((0, 0), color=colormap(i), lw=4) for i in range(len(label_names))]
        self.ax1.legend(custom_legend_entries, label_names, loc='best')

        return xtr, ytr, xts, yts

    def initAxis2(self, lda, title_color):
        pca_loadings = self.pca.components_
        lda_loadings = self.lda.coef_
        loadings = np.dot(lda_loadings, pca_loadings)
        print(pca_loadings.shape, lda_loadings.shape, loadings.shape)

        bins = [float(e) for e in list(self.data_frame[self.data_frame['filename'] == self.point_label].columns)[4:]]
        x = bins #np.linspace(1, loadings.shape[1], loadings.shape[1])
        self.ax2.stem(x, loadings[0, :], 'b')

        # label plot with axis names, title, et cetera
        self.ax2.set_title('Loading Plot', color=title_color)
        self.ax2.set_xlabel('m/z (Daltons)', color=title_color)
        self.ax2.set_ylabel('Magnitude', color=title_color)
        self.ax2.tick_params(axis='x', colors=title_color)
        self.ax2.tick_params(axis='y', colors=title_color)
        self.ax2.grid()

        return x, loadings

    def initAxis3(self, title_color):
        # label plot with axis names, title, et cetera
        self.ax3.set_title('Selected Mass Spectrum', color=title_color)
        self.ax3.set_xlabel('m/z (Daltons)', color=title_color)
        self.ax3.set_ylabel('Intensity (A.U.)', color=title_color)
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

    def projectDataFrames(self, training_file_reader, test_file_reader, lda_training_data, lda_test_data, test_predictions):
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
                'ld2' : lda_training_data[:,1]
                #'ld3' : lda_training_data[:,2]
            })

            if lda_test_data is None:
                test_frame = None
            else:
                test_frame = pd.DataFrame({
                    'index' : test_file_reader.file_frame.iloc[:,0],
                    'label' : 'unknown',
                    'filename' : test_file_reader.file_frame.iloc[:,2],
                    'ld1' : lda_test_data[:,0],
                    'ld2' : lda_test_data[:,1]
                    #'ld3' : lda_test_data[:,2]
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
            }) if file_reader.file_frame.shape[1] > 1 else pd.DataFrame({
                'index' : file_reader.file_frame.iloc[:,0],
                'label' : file_reader.file_frame.iloc[:,1],
                'filename' : file_reader.file_frame.iloc[:,2],
                'ld1' : lda_data[:,0]
            })

    def getLoadings(self):
        return self.loadings

    def getScores(self):
        return self.scores_x, self.scores_y

    def show(self):
        plt.show()

        plt.autoscale(False)

    # cursor hover
    def hover(self, event):
        # check if event was in PCA-LDA plot (axis 1)
        if event.inaxes == self.ax1:    
            # check/get the LDA points contained in the event
            cont1, ind1 = self.sctr.contains(event)
            try:
                cont2, ind2 = self.scts.contains(event)
            except:
                cont2 = False
                ind2 = None
            if cont1:
                self.highlightPCALDA(event, ind1, 1)
            elif cont2:
                self.highlightPCALDA(event, ind2, 2)
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
                self.highlight = self.ax1.plot(float(point['ld1']), float(point['ld2']), marker='o', alpha=0.6, markersize=16, fillstyle='none', color='r')
            except:
                print(f'NOTE: Duplicate points from {self.point_label}')
                point = point.iloc[0,:]
                self.highlight = self.ax1.plot(float(point['ld1']), float(point['ld2']), marker='o', alpha=0.6, markersize=16, fillstyle='none', color='r')


    def highlightPCALDA(self, event, ind, src):
        # change LDA point annotation position
        self.point_annot.xy = (event.xdata, event.ydata)

        # update label to select a single point
        names = self.plot_frame['filename'].values
        if src == 2:
            self.point_label = names[int(ind['ind'][0]) + self.num_training]
        else:
            self.point_label = names[int(ind['ind'][0])]
        
        # reset axis 3
        try:
            for i in range(len(self.ax3.lines)):
                del self.ax3.lines[i]
        except:
            pass

        # get spectral data for selected point
        inten = np.nan_to_num(self.data_frame[self.data_frame['filename'] == self.point_label].values[0][4:])
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
Console Command: python miPlot2D.py <model_data_name.csv> <test_data_name.csv> <pcalda_model.model> <n_std>
Arguments:
    <path/model_data_name.csv> - (String) first CSV file including the extension ".csv"
    <path/test_data_name.csv>  - (String) OPTIONAL - second CSV file which will be plotted and labelled with first CSV file
    <path/pcalda_model.model>  - (String) PCA-LDA model file
    <n_std>                    - (Float) Number of standard deviations that the confidence ellipsoids should extend to"""

def handleStartUpCommands(help_message):
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
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
        print("Type 'python InteractivePlot2D_mpl.py help' for more info")
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

    """
    loadings = plot.getLoadings()
    scores_x, scores_y = plot.getScores()

    fig, axs = plt.subplots(1, 1)
    
    x = np.linspace(100, 1000-1, loadings.shape[1])
    plt.stem(x, loadings[0, :], 'b')
    plt.title('Loading Plot')
    plt.xlabel('m/z')
    plt.ylabel('Magnitude')
    plt.grid()
    plot.initAxis1(axs, 'black', 2)
    
    plt.show()"""

    
    

if __name__ == "__main__":
    main()

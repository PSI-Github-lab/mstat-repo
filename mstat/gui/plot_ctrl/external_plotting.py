try:
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from mstat.dependencies.helper_funcs import *
except ModuleNotFoundError as e:
    import os
    print(f'From {os.path.basename(__file__)}')
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def ext_plot_pcalda_data(self, train=None, test=None, title="PCA Scores", model_classes = None):
    """
    Plot scores data from tuples for training and testing data.
    Tuples contain 
        data: (x, y), labels, known classes
    """

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

        # encode the combined labels
        plot_encoder.fit(labels)

        # create external figure
        ext_fig, ext_axis = plt.subplots(1,1)

        # plot the data
        self.plot_scores_data(data, labels, plot_encoder, model_classes, self.CB_color_cycle, ax=ext_axis)

        # update plot details
        if self.options['legend_checked']:
            u_labels = np.unique(labels)
            custom_legend_entries = [Circle((0, 0), color=self.CB_color_cycle[i], lw=4) for i in plot_encoder.transform(u_labels)]
            legend = ext_axis.legend(custom_legend_entries, u_labels, loc='best')
            legend.set_draggable(True)
        ext_axis.set_title(title)
        ext_axis.set_xlabel(self.options['xaxis_option'])
        ext_axis.set_ylabel(self.options['yaxis_option'])
        ext_fig.tight_layout()
        ext_axis.grid()
        plt.show()
    except Exception as exc:
        print(exc)
        print(f'From {os.path.basename(__file__)}')
        print('Could not plot data')

def ext_plot_loading_data(self, data, title='Loading Plot'):
    """
    data structure (np array, np array)
    (array) m/z range (array) loadings
    """
    # create external figure
    ext_fig, ext_axis = plt.subplots(1,1)

    if data is not None:
        mzs, loadings = data
        self.loading_stem = ext_axis.stem(mzs, loadings)
    
    # label plot with axis names, title, et cetera
    ext_axis.set_title(title)
    ext_axis.set_xlabel('m/z (Daltons)')
    ext_axis.set_ylabel('Magnitude')
    ext_fig.tight_layout()
    ext_axis.grid()
    plt.show()
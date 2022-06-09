# coding: utf-8
from dependencies.ScikitImports import *
import numpy as np
from matplotlib import pyplot as plt, cm
import colorcet as cc
import sys
from datetime import *
import joblib


from dependencies.ms_data.MSFileReader import MSFileReader
from dependencies.ms_data.AnalysisVis import labelled_scatter, labelled_scatter_3d, AnalysisVis
from dependencies.ms_data.MSDataAnalyser import MSDataAnalyser
from dependencies.bootstrapStats import bootstrap_conf
#from dependencies.readModelConfig import *

import numpy as np
import scipy.stats as ss

from scipy.spatial import distance

class Mdist:
    def __init__(self, file_reader, pred_est, do_rand=False) -> None:
        _, _, labels, encoder = file_reader.encodeData()
        self.file_reader = file_reader
        self.data_frame = file_reader.file_frame
        self.encoder = encoder
        self.do_rand = do_rand
        self.classes = np.unique(labels)
        self.pred_est = pred_est
        # get covariance matrix and class means for M dist calculations
        try:
            self.cov = pred_est['lda'].covariance_
            self.mus = pred_est['lda'].means_
            self.inv_cov = np.linalg.inv(self.cov)
            self.da_mode = 'lda'
        except KeyError:
            self.cov = pred_est['qda'].covariance_
            self.mus = pred_est['qda'].means_
            self.inv_cov = np.linalg.inv(self.cov)
            self.da_mode = 'qda'

        print(f' ANALYSIS MODE: {self.da_mode} '.center(80, '*'))

    def getDist(self, data, _class):
        mu = self.mus[_class]
        inv_cov = self.inv_cov if self.da_mode == 'lda' else self.inv_cov[_class]
        pca_data = self.pred_est[:-1].transform(data.reshape(1,-1))

        return distance.mahalanobis(pca_data, mu, inv_cov)

    def getClassDists(self, verbose=False):
        m_dists = []
        for i, label in enumerate(self.classes):
            files = self.data_frame[self.data_frame['label'] == label]['filename']
            dists = []
            for file in files:
                row = self.data_frame[self.data_frame['filename'] == file].values[0]
                data = row[4:] / (self.do_rand and 1.0 or float(row[3]))
                dists.append(self.getDist(data, i))

            lower, upper = bootstrap_conf(dists, diff_or_value=1)
            if verbose:
                print(f"Average M distance for {label} data points - {np.mean(dists)} {lower} {upper}")
           
            m_dists.append([np.mean(dists), lower, upper])
            print(len(dists))
        return self.classes, np.array(m_dists)


help_message = """
Console Command: python PCALDADist.py <model_data_name.csv> <test_data_name.csv> <pcalda_model.model> <data_src>
Arguments:
    <path/model_data_name.csv> - (String) first CSV file including the extension ".csv"
    <path/test_data_name.csv>  - (String) second CSV file which will be plotted and labelled with first CSV file
    <path/pcalda_model.model>  - (String) PCA-LDA model file """

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if not argm:
        print("Type 'python pcaldaDist.py help' for more info")
        quit()
    else:
        first_csv_file = argm[0]
        second_csv_file = argm[1]
        model_file = argm[2]
        do_rand = False#bool(int(argm[3]))
        da_mode = 0#argm[4]

    # get data for analysis (from csv files or generated)
    if not do_rand:
        # load saved pcalda model
        pred_est, _, meta_info = joblib.load(model_file)
        print(f"Loaded model from {model_file}\n")
        try:
            base_pipeline = pred_est.calibrated_classifiers_[0].base_estimator
        except AttributeError:
            base_pipeline = pred_est

        # read data from the csv files
        base_reader = MSFileReader(first_csv_file)
        test_reader = MSFileReader(second_csv_file)

        # load and show data
        _, _, base_labels, base_encoder = base_reader.encodeData()
        print(' DATA FROM CSV FILE '.center(80, '*'))
        print(base_reader)

        _, _, test_labels, test_encoder = test_reader.encodeData()
        print(test_reader)

        # plot data
        colormap = cm.get_cmap('cet_glasbey_light') 
        pca_base_data = base_pipeline[:-1].transform(base_reader.getTICNormalization())
        #if pca_base_data.shape[1] > 2:
        #    ax1 = labelled_scatter_3d(pca_base_data[:,:3], base_labels, colormap)
        #    pca_test_data = base_pipeline[:-1].transform(test_reader.getTICNormalization())
        #    labelled_scatter_3d(pca_test_data[:,:3], test_labels, colormap, ax=ax1)

    else:
        # create random data
        base_reader = MSFileReader('fake', do_rand=True, rand_state=43, rand_centers=[[5,0,0], [0,5,0], [0,15,0]], rand_std=[0.25, 2, 1], rand_name='Model')
        test_reader = MSFileReader('fake', do_rand=True, rand_state=46, rand_centers=[[2.5,2.5,0], [10,5,0], [10,10,0]], rand_std=[0.00005, 0.00005, 0.00005], rand_name='Unknown', rand_num=1)

        _, _, base_labels, base_encoder = base_reader.encodeData()
        print(' RANDOM DATA '.center(80, '*'))
        print(base_reader)

        _, _, test_labels, test_encoder = test_reader.encodeData()
        print(test_reader)

        colormap = cm.get_cmap('rainbow', len(np.unique(test_labels))+len(np.unique(base_labels)))
        f = labelled_scatter(base_reader.file_frame[[1,2]].values, base_labels, colormap, 3, 'Random Data', 'Dim 1', 'Dim 2', base_encoder)
        clm_offset = len(np.unique(base_labels))
        labelled_scatter(test_reader.file_frame[[1,2]].values, test_labels, colormap, 3, '', '', '', test_encoder, f, clm_offset)

        rnd_state = 42
        if da_mode == 'lda':
            # create PCA-LDA estimator
            analysis = MSDataAnalyser('none', 3, rnd_state, da_mode=0)

            analysis.fitModel(base_reader.feature_data, base_labels)
            pred_est = analysis.class_pipeline

            # get covariance matrix and class means for M dist calculations
            cov = pred_est['lda'].covariance_
            mus = pred_est['lda'].means_
            inv_cov = np.linalg.inv(cov)

            print(f'Covariance {cov}') 
            print(f'Means {mus}') 
        else:
            # create PCA-QDA estimator
            analysis = MSDataAnalyser('none', 3, rnd_state, da_mode=1)

            analysis.fitModel(base_reader.feature_data, base_labels)
            pred_est = analysis.class_pipeline

            # get covariance matrice and class means for M dist calculations
            cov = pred_est['qda'].covariance_
            mus = pred_est['qda'].means_
            inv_cov = np.linalg.inv(cov)

        # visualize
        visualisation = AnalysisVis(analysis, base_encoder, '', colormap)
        analysis.transformData()
        _, ax = visualisation.visualisePCA3D(save_fig=False)
        pca_test = pred_est[:-1].transform(test_reader.getTICNormalization())
        labelled_scatter_3d(pca_test, test_labels, colormap, ax, len(np.unique(base_labels)))

    # show user model parameters
    print(' MODEL IN USE '.center(80, '*'))
    print(pred_est)

    # first, show some of the base data results
    print(' DATA FROM MODEL LIBRARY '.center(80, '*'))
    model_dists = Mdist(base_reader, base_pipeline, do_rand=do_rand)
    #classes = model_dists.classes
    classes, m_dists = model_dists.getClassDists(verbose=True)
    #print(f"base: {m_dists}")

    # set up plot for comparing test distances to training distances
    fig = plt.figure(figsize=(16,7))
    #plt.errorbar(classes, m_dists[:,0], yerr=m_dists[:,1:].T, capsize=5, ls='--', label="Model Baseline", c=colormap(0))
    plt.title('Mahalanobis Distance Comparison')
    plt.xlabel('Model Class')
    plt.xticks(rotation = -45, ha='left')
    plt.ylabel('M Distance from Class Mean')
    plt.grid()

    # second, perform same analysis on test data
    print(' UNSEEN DATA '.center(80, '*'))
    test_frame = test_reader.file_frame
    test_classes = np.unique(test_labels)
    #test_classes = [test_classes[0], test_classes[5], test_classes[7]]
    for i, test_label in enumerate(test_classes):
        print(f"Looking at {test_label}")
        files = test_frame[test_frame['label'] == test_label]['filename']
        m_dists = []
        for file in files:
            row = test_frame[test_frame['filename'] == file].values[0]
            data = row[4:] / (do_rand and 1.0 or float(sum(row[4:])))
            proba = pred_est.predict_proba(data.reshape(1,-1))
            #print(f"{file} is predicted as {base_encoder.inverse_transform(prediction)}")
            dists = [
                model_dists.getDist(data, i)
                for i, label in enumerate(
                    np.unique(base_labels)
                )
            ]
            m_dists.append(dists)
        m_dists = np.array(m_dists)

        stats = np.zeros((m_dists.shape[1],3))
        for j, column in enumerate(m_dists.T):
            lower, upper = bootstrap_conf(column, diff_or_value=1)
            stats[j, :] = [np.mean(column), lower, upper]
            
        #print(f"{test_label}: {m_dists}")
        #print(stats)
        if m_dists.shape[0] > 1:
            # if there are many points for each test class then show error bars
            plt.errorbar(x=classes, y=stats[:,0], yerr=stats[:,1:].T, capsize=5, ls='-', label=test_label, c=colormap(i+1))
        else:
            # plot trace for lone samples
            plt.plot(classes, m_dists.T, 'o-', label=test_label, c=colormap(i+1))
            print(f"Raw model confidence: {proba.take(proba.argmax(1))}") # from {proba}")
            n = 80
            p = base_pipeline['dim'].get_params()['n_components']
            x = min(m_dists[0])
            #print(n,p,x)
            obs_cdf = 1 - ss.f.cdf((x**2)*(n*(n-p))/(p*(n-1)*(n+1)), p, n-p)
            print(f"Mdist confidence multiplier: {obs_cdf}")
            print(f"Mdist estimate confidence: {proba.take(proba.argmax(1))*obs_cdf}")

    #plt.yscale('log')
    plt.legend()
    plt.show()      

if __name__ == "__main__":
    main()
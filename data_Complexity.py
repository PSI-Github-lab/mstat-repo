# coding: utf-8
from sklearn.datasets import make_blobs, make_classification, make_moons, make_circles
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import numpy as np
from matplotlib import  pyplot as plt, cm
import sys, os
from datetime import *

from mstat.dependencies.readModelConfig import *
from mstat.dependencies.directory_dialog import *
from mstat.dependencies.helper_funcs import *

from scipy.spatial.distance import pdist
import numpy as np
import scipy.spatial as spatial
import itertools

from scipy.stats import binned_statistic

## https://stats.stackexchange.com/questions/235270/entropy-of-an-image

def calcBins(low, up, bin_size, mz_lims = None):
    if mz_lims is None:
        mz_lims = [-np.inf,np.inf]
    lowlim = max(low, min(mz_lims))
    uplim = min(up, max(mz_lims))
    num_bins = int((uplim - lowlim)/bin_size)
    return np.linspace(lowlim, uplim-bin_size, num_bins), num_bins

def bin_data(mzs, intens, meta, bin_size=1.0):
    #print('BINNING', mzs.shape, intens.shape)
    low = float(meta[0]['lowlim'])
    up = float(meta[0]['uplim'])
    #print(low, up, bin_size)
    bins, num_bins = calcBins(low, up, bin_size)
    rows = []

    for intens_row, mzs_row in zip(intens, mzs):
        stats, bin_edges, _ = binned_statistic(mzs_row, intens_row, 'mean', bins=num_bins, range=(low, up))
        #print(stats[:5], sum(row), sum(stats))
        new_row = np.concatenate(([sum(stats)], stats)) 
        rows.append(stats)

    return bin_edges + bin_size/2, np.array(rows)

def my_entropy(probs):
    n_events = len(probs)
    probs = probs[np.nonzero(probs)]

    if n_events <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_events)

def main():

    # get directories
    dirhandler = DirHandler(log_name='diagpwr', log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    # define all directories to run the coversion
    if 'PREV_SOURCE' in dirs:
        in_directories = getMultDirFromDialog("Choose folders containing npy files for analysis", dirs['PREV_SOURCE'])
    else:
        in_directories = getMultDirFromDialog("Choose folders containing npy files for analysis")
    if len(in_directories) == 0:
        print('Action cancelled. No directories were selected.')
        quit()
    common_src = os.path.commonpath(in_directories)
    dirhandler.addDir('PREV_SOURCE', common_src)

    dirhandler.writeDirs()

    feature_data_arrays = []
    for path in in_directories:
        with open(rf'{path}\{os.path.basename(path)}.npy', 'rb') as f:
            intens = np.load(f, allow_pickle=True)
            mzs = np.load(f, allow_pickle=True)
            meta = np.load(f, allow_pickle=True)

            label = meta[0]['comment1']
            if label == '':
                label = os.path.basename(path)
            
            print(label)
            print(intens.shape)
            bin_edges, binned_intens = bin_data(mzs, intens, meta)
            print(binned_intens.shape)
            feature_data_arrays.append((binned_intens, label))

    feature_data = np.empty((1,feature_data_arrays[0][0].shape[1]))
    labels = np.empty((1,))
    for intens, label in feature_data_arrays:
        feature_data = np.concatenate((feature_data, intens), axis=0)
        labels = np.concatenate((labels, [label]*intens.shape[0]))

    feature_data = feature_data[1:,:]
    labels = labels[1:]

    #a = a[0]
    #selection_mask = np.logical_and((a >= low_lim),(a <= up_lim))
    #a, b = a[selection_mask], b[selection_mask]
    
    #avg_spectrum = avg_spectrum / np.max(avg_spectrum)
    feature_data = getTICNormalization(feature_data)
    avg_spectrum = np.mean(feature_data, axis=0)

    print(feature_data.shape, labels.shape)

    #X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = feature_data, feature_data, labels, labels

    # for now X_train will be all of the feature data
    pca = PCA().fit(X_train)
    percent_variance = pca.explained_variance_ratio_
    #print(percent_variance)
    threshold = 0.95

    n = X_train.shape[0]
    for i in range(1, len(percent_variance)):
        if sum(percent_variance[:i]) >= threshold:
            n = i
            #print(i, i / X_train.shape[0], sum(percent_variance[:i]))
            print(f"{threshold * 100}% variance captured with {i} principal components")
            print(f"PCA complexity score of {np.log(i / X_train.shape[1])}")
            #print(percent_variance[:i])
            break

    print(avg_spectrum.shape)

    #print(np.histogram(avg_spectrum, bins='auto'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(avg_spectrum)
    didm = np.gradient(avg_spectrum) #np.diff(avg_spectrum) / np.diff(bin_edges[:-1])
    diffRange = np.max( [ np.abs( didm.min() ), np.abs( didm.max() ) ] )
    ax.plot(didm)
    plt.show()

    fig1, ax1 = plt.subplots(1, 1)

    '''num_bins = 1000
    hist_vals, bins, _ = plt.hist(avg_spectrum, bins=num_bins)
    print(hist_vals, sum(hist_vals))

    print(entropy(hist_vals / sum(hist_vals)) / np.log(num_bins))
    print(my_entropy(hist_vals / sum(hist_vals)))'''

    num_bins = 1024
    hist_vals, bins = np.histogram(didm, bins=num_bins, range=[-diffRange, diffRange])
    print(hist_vals, sum(hist_vals))

    print('H =', 0.5*entropy(hist_vals / sum(hist_vals), base=2)) # / np.log(num_bins))

    #plt.show()


    '''#n = 12

    steps = [
        ('dim', PCA(n_components=n, random_state=0)), #n_components=n, 
        ('lda', LDA(store_covariance=True)),
        ]
    pred_est = Pipeline(steps)

    pred_est.fit(X_train, y_train)
    #pca = PCA().fit(feature_data)
    #pca_data = pca.transform(feature_data)
    #lda = LDA().fit(pca_data, labels)
    print(pred_est)

    result = permutation_importance(pred_est, X_test, y_test, n_repeats=10, random_state=0)
    print(result.importances_mean)
    
    fig, ax = plt.subplots()
    ax.bar(np.arange(1, result.importances_mean.shape[0]+1), result.importances_mean, align='center', alpha=0.5, ecolor='black', capsize=5) #yerr=result.importances_std,
    plt.show()'''

    '''corr_matrix = np.corrcoef(feature_data, rowvar=False)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.abs(corr_matrix))
    plt.show()'''

if __name__ == "__main__":
    main()

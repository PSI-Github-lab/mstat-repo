# coding: utf-8
from sklearn.datasets import make_blobs, make_classification, make_moons, make_circles
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
from matplotlib import  pyplot as plt, cm
import sys, os
from datetime import *
#from sklearn.preprocessing import StandardScaler, Normalizer

from mstat.dependencies.ms_data.MSFileReader import MSFileReader
from mstat.dependencies.readModelConfig import *
from mstat.dependencies.directory_dialog import *
from mstat.dependencies.helper_funcs import *

from scipy.spatial.distance import pdist
import numpy as np
import scipy.spatial as spatial
import itertools

def unique_combinations(elements):
    """
    Precondition: `elements` does not contain duplicates.
    Postcondition: Returns unique combinations of length 2 from `elements`.

    >>> unique_combinations(["apple", "orange", "banana"])
    [("apple", "orange"), ("apple", "banana"), ("orange", "banana")]
    """
    return list(itertools.combinations(elements, 2))

def compute_average_distance(X):
    """
    Computes the average distance among a set of n points in the d-dimensional space.

    Arguments:
        X {numpy array} - the query points in an array of shape (n,d), 
                          where n is the number of points and d is the dimension.
    Returns:
        {float} - the average distance among the points
    """
    return np.mean(pdist(X))

help_message = """
Console Command: python DiagPower.py <path/file_name.csv>
Arguments:
    <path/file_name.csv>    - (String) path and name of CSV file including the extension ".csv"
    """

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def get_dataset(mode, dim):
    if mode == 0:
        X, y = make_blobs(
                        n_samples=[100, 100],
                        n_features=dim,
                        cluster_std=[1,0.25],
                        center_box=(-.15,.15),
                        shuffle=False
                            )
    elif mode == 1:
       X, y = make_classification(
                        n_samples=200, 
                        n_features=dim, 
                        n_redundant=0, 
                        n_informative=dim,
                        n_clusters_per_class=1, 
                        n_classes=2,
                        hypercube=False,
                        class_sep= 1,
                        flip_y=0.0
                        #scale=0.1
                            )
	
    return X, y

def calcFDR(class1, class2):
    prev = 0
    for feature in range(class1.shape[1]):
        mu1 = np.mean(class1[:, feature])
        mu2 = np.mean(class2[:, feature])
        var1 = np.var(class1[:, feature])
        var2 = np.var(class2[:, feature])

        val = ((mu1 - mu2)**2) / (var1**2 + var2**2) if var1**2 + var2**2 > 0 else 0
        if val > prev:
            prev = val

    return prev

def calcDK(class1, class2):
    tree1 = spatial.cKDTree(class1)
    tree2 = spatial.cKDTree(class2)

    dists1, _ = tree1.query(class1, 2)
    d1 = 2*np.sqrt(np.mean(dists1**2))
    #print(dists1[:, 1:], d1)
    dists2, _ = tree2.query(class2, 2)
    d2 = 2*np.sqrt(np.mean(dists2**2)) #*np.mean(dists2[:, 1:])
    #print(dists[:, 1], d2)

    s, o = 0, 0

    for i in range(len(class1)):
        c_point = class1[i]

        s += len(tree1.query_ball_point(c_point, d1))-1
        o += len(tree2.query_ball_point(c_point, d1))

        #print(c_point, tree1.query_ball_point(c_point, d1), tree2.query_ball_point(c_point, d1))

    for i in range(len(class2)):
        c_point = class2[i]

        s += len(tree2.query_ball_point(c_point, d2))-1
        o += len(tree1.query_ball_point(c_point, d2))

        #print(c_point, tree1.query_ball_point(c_point, d2), tree2.query_ball_point(c_point, d2))

    return (s - o) / (s + o)

def diagPower(feature_data, labels):
    u_labels = np.unique(labels)
    print(u_labels)
    # calculate fisher discriminant ratio
    FDR = calcFDR(feature_data[(labels==u_labels[0]),:], feature_data[(labels==u_labels[1]),:])
    # calculate dong-kathari coef
    DKC = -1
    ind = -1
    for i in range(2, min(feature_data.shape[0], feature_data.shape[1])+1):
        data = PCA(n_components=i).fit_transform(feature_data)
        nDKC = calcDK(data[(labels==u_labels[0]),:], data[(labels==u_labels[1]),:])
        #print(nDKC)
        if nDKC > DKC:
            DKC = nDKC
            ind = i
        if DKC == 1.000:
            break
    # calculate silhouette_score
    SC = metrics.silhouette_score(feature_data, labels)

    #print(f"Fisher Discriminant Ratio: {FDR}")
    #print( """Fisher Discr. Ratio:  {:.2e}""".format(FDR))
    print("""Dong-Kothari Coef:    {:.3f} {:.3f} {:.3f}""".format(DKC, ind, calcDK(feature_data[(labels==u_labels[0]),:], feature_data[(labels==u_labels[1]),:])))
    print("""Silhouette Coef:      {:.5f}""".format(SC))

    return FDR, DKC, SC

def datasetDP(feature_data, labels):
    u_labels = np.unique(labels)
    FDRs = []
    DKCs = []
    SCs = []
    if len(u_labels) <= 2:
        loo = LeaveOneOut()
        loo.get_n_splits(feature_data)

        for train_index, test_index in loo.split(feature_data):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = feature_data[train_index], feature_data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            F, D, S = diagPower(X_train, y_train)
            FDRs.append(F); DKCs.append(D); SCs.append(S)
        return FDRs, DKCs, SCs
    
    for pair in unique_combinations(u_labels):
        print(pair)
        n1 = len(feature_data[(labels==pair[0]),:])
        n2 = len(feature_data[(labels==pair[1]),:])
        data = np.concatenate((feature_data[(labels==pair[0]),:], feature_data[(labels==pair[1]),:]))

        F, D, S = diagPower(data, np.array([pair[0]]*n1 + [pair[1]]*n2))
        FDRs.append(F); DKCs.append(D); SCs.append(S)
    return FDRs, DKCs, SCs

from scipy.stats import binned_statistic
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

    for row in intens:
        stats, bin_edges, _ = binned_statistic(mzs, row, 'sum', bins=num_bins, range=(low, up))
        #print(stats[:5], sum(row), sum(stats))
        new_row = np.concatenate(([sum(stats)], stats)) 
        rows.append(stats)

    return bin_edges + bin_size/2, np.array(rows)

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if not argm:
        cont_check = True
        file_name = None
    else:
        file_name = argm[0]

    if file_name == 'rand' and not cont_check:
        n = 1801
        feature_data, labels = get_dataset(0, n)
        print(f' RANDOM {n}-DIMENSIONAL DATA '.center(80, '*'))
        #feature_data = PCA(n_components=2).fit_transform(feature_data)
        diagPower(feature_data, labels)

        clm = cm.get_cmap('rainbow', len(np.unique(labels)))
        plt.scatter(feature_data[:,0], feature_data[:,1], color=clm(labels))
        plt.grid()

    elif file_name == 'test' and not cont_check:
        fig, axes = plt.subplots(2, 3,
                        figsize=(16, 8))

        same_features, same_labels = make_classification(
                        n_samples=200, 
                        n_features=2, 
                        n_redundant=0, 
                        n_informative=2,
                        n_clusters_per_class=1, 
                        n_classes=1,
                        hypercube=False,
                        class_sep= 0,
                        flip_y=0.0,
                        random_state=9
                        #scale=0.1
                            )

        same_labels[100:] = [1]*100

        datasets = [
            make_classification(
                        n_samples=200, 
                        n_features=2, 
                        n_redundant=0, 
                        n_informative=2,
                        n_clusters_per_class=1, 
                        n_classes=2,
                        hypercube=False,
                        class_sep= 50,
                        flip_y=0.0,
                        random_state=0
                        #scale=0.1
                            ),
                    make_classification(
                        n_samples=200, 
                        n_features=2, 
                        n_redundant=0, 
                        n_informative=2,
                        n_clusters_per_class=1, 
                        n_classes=2,
                        hypercube=False,
                        class_sep= 7,
                        flip_y=0.0,
                        random_state=5
                        #scale=0.1
                            ),
                    make_classification(
                        n_samples=200, 
                        n_features=2, 
                        n_redundant=0, 
                        n_informative=2,
                        n_clusters_per_class=1, 
                        n_classes=2,
                        hypercube=False,
                        class_sep= 3,
                        flip_y=0.0,
                        random_state=5
                        #scale=0.1
                            ),
                    (same_features, same_labels),
                    make_moons(n_samples=200, noise=0.2),
                    make_circles(n_samples=200, noise=0.1, factor=0.6)
                            ]

        names = [
            'Well Separated Clusters', 'Close Clusters', 'Overlapping Clusters', 
            'Same Clusters', 'Moons', 'Circles'
        ]

        for dataset, name, i in zip(datasets, names, range(len(names))):
            print(f' {name} '.center(80, '*'))
            feature_data, labels = dataset

            #feature_data = PCA(n_components=2).fit_transform(feature_data)
            FDR, DKC, SC = diagPower(feature_data, labels)

            textstr = '\n'.join((
                    r'$Max Sep={:.2e}$'.format(np.mean(FDR)),
                    r'$Overlap=%.3f$' % (DKC, ),
                    r'$Quality=%.3f$' % (SC, )))

            ax = plt.subplot(2,3,i+1)
            ax.set_title(f'{name}', size=16)
            ax.grid()
            clm = cm.get_cmap('rainbow', len(np.unique(labels)))
            ax.scatter(feature_data[:,0], feature_data[:,1], color=clm(labels))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

    else:
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
                _, binned_intens = bin_data(mzs, intens, meta)
                print(binned_intens.shape)
                feature_data_arrays.append((intens, label))

        feature_data = np.empty((1,feature_data_arrays[0][0].shape[1]))
        labels = np.empty((1,))
        for intens, label in feature_data_arrays:
            feature_data = np.concatenate((feature_data, intens), axis=0)
            labels = np.concatenate((labels, [label]*intens.shape[0]))

        feature_data = feature_data[1:,:]
        labels = labels[1:]

        feature_data = getTICNormalization(feature_data)
        #feature_data = PCA(n_components=10).fit_transform(feature_data)

        print(feature_data.shape, labels.shape)

        FDR, DKC, SC = datasetDP(feature_data, labels)

        print('LeaveOneOut CV Results'.center(80, '*'))
        dkc_m, dkc_ci = mean_normal_cinterval(DKC, confidence=0.95)
        dkc_std = np.std(DKC, ddof=1)
        sc_m, sc_ci = mean_normal_cinterval(SC, confidence=0.95)
        sc_std = np.std(SC, ddof=1)
        textstr = '\tDKC={:.3f}+-{:.3f}'.format(dkc_m, 2*dkc_std) + '\tSC={:.3f}+-{:.3f}'.format(sc_m, 2*sc_std)
        print(textstr)
        print(r'For more information see C:\Users\Jackson\PSI Files Dropbox\Slides\JR\2021-09-07_DiagnosticPower.pptx', '\n')
        input('Press ENTER to leave script...')
        quit()

if __name__ == "__main__":
    main()

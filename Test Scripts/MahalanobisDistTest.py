# coding: utf-8
from operator import length_hint
import scipy
from sklearn.datasets import make_blobs, make_classification, make_moons, make_circles
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, learning_curve, StratifiedKFold, validation_curve, KFold
from sklearn.metrics import fbeta_score, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.covariance import EmpiricalCovariance, MinCovDet, LedoitWolf
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
from matplotlib import projections, pyplot as plt, cm
from matplotlib.patches import Circle
import sys, os
from datetime import *

from dependencies.ms_data.MSFileReader import MSFileReader
from dependencies.ms_data.AnalysisVis import AnalysisVis
from dependencies.ms_data.MSDataAnalyser import MSDataAnalyser
from readModelConfig import *

from scipy.spatial.distance import pdist
from scipy.stats import gmean
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.spatial as spatial
import itertools

from scipy.spatial import distance

# get the dataset
def get_dataset(mode, dim):
    if mode == 0:
        X, y = make_blobs(
                        n_samples=[1000, 1000, 1000],
                        n_features=dim,
                        cluster_std=[0.5, 0.5, 0.5],
                        center_box=(-10,10),
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

def main():
    X, y = get_dataset(0, 3)

    lda = LDA(store_covariance=True)
    lda.fit(X, y)

    print(lda.covariance_)
    iv = np.linalg.inv(lda.covariance_)
    print(iv)

    print(f"[1, 0, 0], [0, 1, 0]: {distance.mahalanobis([1, 0, 0], [0, 1, 0], iv)}")
    print(f"[0, 2, 0], [0, 1, 0]: {distance.mahalanobis([0, 2, 0], [0, 1, 0], iv)}")
    print(f"[2, 0, 0], [0, 1, 0]: {distance.mahalanobis([2, 0, 0], [0, 1, 0], iv)}")

if __name__ == "__main__":
    main()
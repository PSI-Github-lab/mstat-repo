import numpy as np
import scipy.spatial as spatial
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import LeaveOneOut

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
    d1 = np.median(dists1[:, 1:])
    #print(dists1[:, 1:])
    #print(dists1.shape)
    #print('data', np.median(dists1[:, 1:], axis=0), np.median(dists1[:, 1:], axis=1), np.mean(dists1[:, 1:], axis=0), np.mean(dists1[:, 1:], axis=1))
    dists2, _ = tree2.query(class2, 2)
    d2 = np.median(dists2[:, 1:]) #*np.mean(dists2[:, 1:])
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

    return abs((s - o) / (s + o))

def calcPCAComplexity(feature_data, threshold=0.95):
    pca = PCA().fit(feature_data)
    percent_variance = pca.explained_variance_ratio_
    #print(percent_variance)

    for i in range(1, len(percent_variance)):
        if sum(percent_variance[:i]) >= threshold:
            #print(f"{threshold * 100}% variance captured with {i} principal components")
            PC = np.log(i / feature_data.shape[1])
            #print(f"Classification complexity score of {PC}")
            return PC
    return 0.0

def calc1NNError(feature_data, labels):
    loo = LeaveOneOut()
    loo.get_n_splits(feature_data)

    incorr_pred = []
    for train_index, test_index in loo.split(feature_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = feature_data[train_index], feature_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        NNC = KNN(n_neighbors=1).fit(X_train, y_train)
        prediction = NNC.predict(X_test)
        #print(prediction, y_test)
        incorr_pred.append(prediction[0] != y_test[0])

    return sum(incorr_pred) / len(incorr_pred)
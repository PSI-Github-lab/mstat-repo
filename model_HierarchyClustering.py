# coding: utf-8
try:
    import numpy as np
    import sys
    from matplotlib import cm, pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial import KDTree
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
    from mstat.dependencies.ScikitImports import *
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

np.set_printoptions(precision=5, suppress=True)

def convHier(data, encoder):
    n = len(encoder.classes_)
    if data >= n:
        return chr(2*n - 2 - data + 65)

    return encoder.inverse_transform([data])[0]
vconvHier = np.vectorize(convHier)

def getHier(data, encoder):
    linkage_matrix = linkage(data, 'single')
    data = linkage_matrix[:, :2].astype(int)
    levels = np.flipud(vconvHier(data, encoder))

    keys = [chr(x + 65) for x in range(14)]
    keys[0] = 'ROOT'

    return dict(zip(keys, levels)), linkage_matrix
    

def plot_dendrogram(linkage_matrix, **kwargs):
    #c, coph_dists = cophenet(linkage_matrix, pdist(X))

    # create the counts of samples under each node
    '''
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)'''

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def prob_dendrogram(probs, *args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Classifier Dendrogram')
        plt.xlabel('Class')
        plt.ylabel('Separation')

        dd = list(np.array(ddata['dcoord'])[:,1])
        ii = ddata['icoord']
        prob_colormap = cm.get_cmap('brg')
        for j, (d, i) in enumerate(sorted(zip(dd, ii), reverse=True)):
            y = d
            if probs[j] >= 0:
                x = 0.5 * sum(i[1:3])
                plt.plot(x, y, 'o', c=prob_colormap(0.4 + probs[j]/2), markersize=50*(0.5 + probs[j]/3))
                plt.annotate("%.2f" % probs[j], (x, y), xytext=(0, 4),
                                textcoords='offset points',
                                va='top', ha='center', color='white', weight='bold' if probs[j] > 0.5 else 'normal')
        if max_d:
            plt.axhline(y=max_d, c='k')
        plt.yticks([])
    return list(ddata['leaves'])

help_message = """
Console Command: python HierarchyClustering.py <model_data_name.csv>
Arguments:
    <path/file_name.csv> - (String) CSV file including the extension ".csv" """

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
        print("Type 'python HierarchyClustering.py help' for more info")
        quit()
    else:
        csv_file = argm[0]
    
    # data from csv file
    training_reader = MSFileReader(csv_file)
    _, training_data, training_labels, encoder = training_reader.encodeData()
    print(training_reader)

    training_data = training_reader.getTICNormalization()
    #training_data = PowerTransformer().fit_transform(training_data)
    n = len(encoder.classes_)
    X = np.empty((n, training_data.shape[1]))

    for i, label in enumerate(np.unique(training_labels)):
        class_data = training_data[(training_labels == label)]

        mean = np.mean(class_data, axis=0)
        X[i] = mean
    print(X.shape)

    
    # use relative distances of data to create clustering threshold
    tree = KDTree(X)
    nearest_dist, nearest_ind = tree.query(X, k=2)
    cust_thresh = nearest_dist[0,1]

    print(f"Nearest sep: {nearest_dist[0,1]}")
    '''
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single')

    model = model.fit(X)
    print(model.labels_)'''

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    hdict, linkage_matrix = getHier(X, encoder)
    plot_dendrogram(linkage_matrix, truncate_mode="level", p=n-1, labels=encoder.classes_, leaf_rotation=90)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.yticks([])

    plt.figure()
    colormap = cm.get_cmap('cet_glasbey_light') 
    plt.scatter(X[:,0], X[:,1])#, c=colormap(model.labels_))
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
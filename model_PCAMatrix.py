# coding: utf-8
try:
    import sys
    import numpy as np
    from matplotlib import pyplot as plt
    from mstat.dependencies.ScikitImports import *
    from mstat.dependencies.ms_data.MSFileReader import MSFileReader
    from mstat.dependencies.directory_dialog import *
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def pcaCompare(feature_data, labels, classes, tc):
    f = np.logical_or((labels == tc), (labels == classes[4]))
    n = int(np.ceil(0.2 * feature_data[f].shape[0]))
    pca = PCA(n_components=n).fit(feature_data[f])
    for c in classes:
        data = pca.transform(feature_data[labels == c])

        if c == tc or c == classes[4]:
            plt.scatter(data[:,0], data[:,1], marker='x')
        else:
            plt.scatter(data[:,0], data[:,1], marker='o')

    #plt.annotate(list(range(len(data))), (data[:,0], data[:,1]))

    plt.show()
    
help_message = """
Console Command: python PCAMatrix.py
Arguments:
    NONE
     """

def handleStartUpCommands(help_message):
    """ extract individual arguments from command line input
        then return arguments in an array"""
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
    dirhandler = DirHandler(log_name='pcamatrix', log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    if 'PREV_TARGET' in dirs:
        train_file_name = getFileDialog("Choose csv data", dirs['PREV_TARGET'])
    else:
        train_file_name = getFileDialog("Choose csv data")
    if len(train_file_name) == 0:
        print('Action cancelled. No directory selected.')
        quit()
    dirhandler.addDir('PREV_TARGET', os.path.dirname(train_file_name))

    dirhandler.writeDirs()
    
    # read training data from the csv file
    reader = MSFileReader(train_file_name)
    frame, feature_data, labels, encoder = reader.encodeData()
    print(reader)
    classes = np.unique(labels)
    print(classes)

    tc = classes[0]


    pcaCompare(feature_data, labels, classes, tc)

    #create grid of PCA score plots
    for a in classes:
        for b in classes:
            # train the pca model
            if a != b:
                n = int(np.ceil(0.2 * feature_data[labels == a].shape[0]))
                pca = PCA(n_components=n).fit(feature_data[labels == a])
                data = pca.transform(feature_data[labels != a])
                print(data.shape)
            else:
                print('pass')

if __name__ == "__main__":
    main()

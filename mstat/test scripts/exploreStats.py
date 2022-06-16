# coding: utf-8
import numpy as np
#from matplotlib import projections, pyplot as plt
import matplotlib.pyplot as plt
import sys, os
from datetime import *
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from dependencies.ms_data.MSFileReader import MSFileReader

help_message = """
Console Command: python exploreStats.py <path/file_name.csv>
Arguments:
    <path/file_name.csv> - (String) path and name of CSV file including the extension ".csv"
    <plots>              - (String) list of plots from 'box', 'hist'
    """

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
        print("Type 'python pcalda.py help' for more info")
        quit()
    else:
        file_name = argm[0]
        plots = argm[1]

    # read data from the csv file
    file_reader = MSFileReader(file_name)
    feature_data, labels, encoder = file_reader.encodeData()
    print(' DATA FROM CSV FILE '.center(80, '*'))
    print(file_reader)

    print(' CSV DATA STATS '.center(80, '*'))
    scl = Pipeline([('a', RobustScaler())])
    corr, bins, df = file_reader.describeData(scl, 200, 300)

    if 'hist' in plots:
        df.hist(bins=100)
    if 'box' in plots:
        df.boxplot()

    plt.matshow(corr)

    step = 50
    plt.xticks(range(0, len(bins), step), bins[::step], fontsize=8, rotation=45)
    plt.yticks(range(0, len(bins), step), bins[::step], fontsize=8)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=12)
    plt.show()

if __name__ == "__main__":
    main()
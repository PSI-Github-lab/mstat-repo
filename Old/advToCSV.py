import sys
import pandas as pd
from subprocess import *
import os.path
import time
from sklearn.model_selection import train_test_split

from dependencies.file_conversion.BatchFile import BatchFile
from dependencies.file_conversion.MZMLDirectory import MZMLDirectory

import time
import numpy as np
from scipy.stats import binned_statistic

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def calcBins(lowlim, uplim, bin_size):
    num_bins = int((uplim - lowlim)/bin_size)+1
    return np.linspace(lowlim, uplim, num_bins)

help_message = """
Console command: python advToCSV.py <path/to/RAW/files> <CSV name> <bin size> <run RAW conversion>
Arguments:
    <paths/to/CSV/files> - (String) directories with RAW data. Paths can be relative or absolute.
    <new CSV name>           - (String) name of the CSV file output. No need to add '.csv' at the end.
    <bin size>           - (Float) bin size for adding all scans together in each RAW file.
    <min mass>           - (Float) max m/z value to consider. Constrained by m/z range observed in RAW files.
    <max mass>           - (Float) min m/z value to consider. Constrained by m/z range observed in RAW files."""

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)

    # define workign directory
    if len(argm) == 0:
        print("ERROR: Please provide arguments when calling the script. Type 'python rawToCSV.py help' for more information.")
        quit()
   
    directories = []
    for arg in argm[:]:
        if '\\' not in arg and '/' not in arg:
            break
        directories.append(arg.replace('/', '\\'))
        argm.remove(arg)

    csv_name = argm[0]
    bin_size = float(argm[1])
    low_lim = float(argm[2])
    up_lim = float(argm[3])
    
    print(' STARTING FILE COLLECTION '.center(80, '*'))

    ''' run batch file for converting RAW files to mzML format for each directory'''
    frames = []
    bins = calcBins(low_lim, up_lim, bin_size)
    first_row = np.concatenate((['label', 'filename', 'total'], [str(e) for e in bins]))
    for directory in directories:
        start_time = time.time()
        print(f"""STATUS: Moved to {directory}\n""")
        # iterate through all the csv files
        l = len(os.listdir(directory))
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        frame = None
        data = []
        for i, file in enumerate(os.listdir(directory)):
            if file.endswith('csv'):
                # read csv and skip first row
                data_frame = pd.read_csv(directory + '/' + file, skiprows=[0], low_memory=False)
                # remove all extra meta data (removing all rows after the first empty row)
                try:
                    scan_frame = data_frame.iloc[:([i for i, x in enumerate(data_frame.iloc[:,1].isna()) if x][0])]
                except:
                    scan_frame = data_frame
                # get scans from scan_frame
                file_bins = scan_frame.columns[1:].astype(float)
                scans = scan_frame.iloc[:,1:].astype(float).values
                # mean of every column from scans
                avg_scans = np.mean(scans, axis=0)
                
                bin_means = binned_statistic(file_bins, avg_scans, bins=len(bins), range=(low_lim, up_lim))[0]
                bin_means[np.isnan(bin_means)] = 0
                
                # get total TIC
                sum_scan = sum(bin_means)

                data.append(np.concatenate(([directory, file, sum_scan], bin_means)))
            
            printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        frame = pd.DataFrame(data=data, columns=first_row)
        frames.append(frame)

        print(f" Ran in {time.time() - start_time} seconds ".center(80, '-'))   

    ''' combine to single csv '''
    print(' COMBINING DATA IN SINGLE CSV FILE '.center(80, '*'))
    start_time = time.time()

    final_data = pd.concat(frames)
    final_data.to_csv('csv_output/' + csv_name + '.csv', index=True)
    
    print(f" Ran in {time.time() - start_time} seconds ".center(80, '-'))

    print("STATUS: CSV files written. Program completed successfully!")
        

if __name__ == '__main__':
    main()
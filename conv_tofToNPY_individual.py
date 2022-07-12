import sys
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from subprocess import *
import os.path
import time
from matplotlib import pyplot as plt

from mstat.dependencies.helper_funcs import get_num_files
from mstat.dependencies.directory_dialog import DirHandler, getMultDirFromDialog

def calcBins(lowlim, uplim, bin_size):
    num_bins = int((uplim - lowlim)/bin_size)+1
    return np.linspace(lowlim, uplim, num_bins)

def main():
    multi_dir_log = 'advnpy'

    # define working directory
    dirhandler = DirHandler(log_name=multi_dir_log, log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    # define all directories to run the coversion
    if 'PREV_SOURCE' in dirs:
        in_directories = getMultDirFromDialog("Choose folders containing raw files for conversion", dirs['PREV_SOURCE'])
    else:
        in_directories = getMultDirFromDialog("Choose folders containing raw files for conversion")
    if len(in_directories) == 0:
        print('Action cancelled. No directories were selected.')
        quit()
    common_src = os.path.commonpath(in_directories)
    dirhandler.addDir('PREV_SOURCE', common_src)

    dirhandler.writeDirs()
    
    print(' STARTING FILE COLLECTION '.center(80, '*'))
    start_time = time.time()

    ''' run batch file for converting RAW files to mzML format for each directory'''
    for directory in in_directories:
        if get_num_files(directory, '.csv') == 0:
            #raise ValueError(f'No data files found in {os.path.basename(directory)}')
            print(f'No data files found in {os.path.basename(directory)}')
        print('Viewing:', os.path.basename(directory))

        start_time = time.time()

        mzs = []
        intens = []
        metadata = []
        for file in tqdm(os.listdir(directory), desc='Processed files: ', total=len(os.listdir(directory))):
            if file.endswith('csv'):
                # read csv and skip first row 
                data_frame = pd.read_csv(directory + '/' + file, low_memory=False)
                # remove all extra meta data (removing all rows after the first empty row)
                # get scans from scan_frame
                #print(data_frame)
                file_bins = data_frame.columns[4:].astype(float)
                #scans = scan_frame.iloc[:,4:].astype(float).values

                for index, row in data_frame.iterrows():
                    #print(row.values, np.array([row.values[4:]]).shape)
                    meta = {
                        'filename' : row.values[2], 
                        'comment1' : row.values[1], 
                        'comment2' : '',    #scan_frame.iloc[0,2] 
                        'lowlim' : str(file_bins.min()),
                        'uplim' : str(file_bins.max()),
                    }

                    
                    # uncomment for TOF data
                    file_name = rf'{directory}\{row[2]}.npy'
                    with open(file_name, 'wb') as f:
                        np.save(f, np.array([row.values[4:].astype(float)]))
                        np.save(f, np.array([file_bins]))
                        np.save(f, np.array([meta]))

                    metadata.append(meta)

        print(f"--- completed in {time.time() - start_time} seconds ---")

        #print('Bins Shape', mzs.shape, 'Counts Shape', intens.shape)
        #print(metadata)
    
    print(f"All ran in {time.time() - start_time} seconds ".center(80, '-'))

    print("STATUS: NPY files written. Program completed successfully!")
        

if __name__ == '__main__':
    main()
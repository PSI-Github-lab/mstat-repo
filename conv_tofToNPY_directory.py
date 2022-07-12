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
                try:
                    scan_frame = data_frame.iloc[:([i for i, x in enumerate(data_frame.iloc[:,1].isna()) if x][0])]
                except:
                    scan_frame = data_frame
                # get scans from scan_frame
                #print(scan_frame)
                file_bins = scan_frame.columns[4:].astype(float)
                scans = scan_frame.iloc[:,4:].astype(float).values
                sum_scans = np.sum(scans, axis=0)

                mzs.append(file_bins)
                intens.append(sum_scans)

                meta = {
                    'filename' : str(os.path.basename(file)), 
                    'comment1' : scan_frame.iloc[0,1], 
                    'comment2' : '',    #scan_frame.iloc[0,2] 
                    'lowlim' : str(file_bins.min()),
                    'uplim' : str(file_bins.max()), 
                    'numscans' : str(scans.shape[0])
                }

                
                # uncomment for TOF data
                file_name = rf'{directory}\{os.path.basename(directory)}.npy'
                with open(file_name, 'wb') as f:
                    np.save(f, np.array(scans))
                    np.save(f, np.array(scans.shape[0] * [file_bins]))
                    np.save(f, np.array(scans.shape[0] * [meta]))

                metadata.append(meta)

        print(f"--- completed in {time.time() - start_time} seconds ---")

        #print('Bins Shape', mzs.shape, 'Counts Shape', intens.shape)
        #print(metadata)
    
    print(f"All ran in {time.time() - start_time} seconds ".center(80, '-'))

    print("STATUS: NPY files written. Program completed successfully!")

    figure = plt.figure()

    i = 0
    plt.scatter(mzs[i], intens[i])
    plt.title(os.path.basename(metadata[i]['filename']))
    plt.grid()
    plt.xlabel('m/z (Da)')
    plt.ylabel('counts (A.U.)')

    plt.show()
        

if __name__ == '__main__':
    main()
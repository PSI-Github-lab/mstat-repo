import numpy as np
import time
from matplotlib import pyplot as plt
from mstat.dependencies.helper_funcs import *
from mstat.dependencies.file_conversion.QuadCConversion import quadc_to_numpy_matrix
from mstat.dependencies.directory_dialog import *

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def main():
    file_log = 'plotnpy'

    # define working directory
    dirhandler = DirHandler(log_name=file_log, log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    # get file from dialog
    if 'PREV_SOURCE' in dirs:
        in_dir = getFileDialog("Choose a file", "NPY files (*.npy)|*.npy", dirs['PREV_SOURCE'])
    else:
        in_dir = getFileDialog("Choose a file", "NPY files (*.npy)|*.npy")
    if len(in_dir) == 0:
        print('Action cancelled. No directories were selected.')
        quit()
    dirhandler.addDir('PREV_SOURCE', in_dir)

    dirhandler.writeDirs()

    #start_time = time.time()

    with open(in_dir, 'rb') as f:
        intens_matrix = np.load(f, allow_pickle=True)
        mzs_matrix = np.load(f, allow_pickle=True)
        meta_list = np.load(f, allow_pickle=True)


    #print(f"--- completed in {time.time() - start_time} seconds ---")
    mode = 'avg'
    figure = plt.figure()
    i = 0
    bin_size = 0.5
    if mode == 'ind':
        for intens, mzs, metadata in zip(intens_matrix, mzs_matrix, meta_list):
            print('Bins Shape', mzs.shape, 'Counts Shape', intens.shape)
            print('m/z limits', metadata['lowlim'], metadata['uplim'])
            i += 1
            low_lim = float(metadata['lowlim'])
            up_lim = float(metadata['uplim'])
            bins, num_bins = calcBins(low_lim, up_lim, bin_size)
            stats, bin_edges, _ = binned_statistic(mzs, intens / np.sum(intens), 'sum', bins=num_bins, range=(low_lim, up_lim))

            stats[np.isnan(stats)] = 0
            #if i > 0 and i < 45:
            plt.plot(bin_edges[:-1] + bin_size/2, (i*0.01) + stats, label='Binned Spectral Sum')
        plt.title(os.path.basename(metadata['filename']))
    elif mode == 'avg':
        stat_total = None
        for intens, mzs, metadata in zip(intens_matrix, mzs_matrix, meta_list):
            print('Bins Shape', mzs.shape, 'Counts Shape', intens.shape)
            print('m/z limits', metadata['lowlim'], metadata['uplim'])
            i += 1
            
            low_lim = float(metadata['lowlim'])
            up_lim = float(metadata['uplim'])
            bins, num_bins = calcBins(low_lim, up_lim, bin_size)
            if stat_total is None:
                stat_total = np.zeros(bins.shape)
            stats, bin_edges, _ = binned_statistic(mzs, intens / np.sum(intens), 'sum', bins=num_bins, range=(low_lim, up_lim))
            stats[np.isnan(stats)] = 0
            stat_total += stats
            
        plt.plot(bin_edges[:-1] + bin_size/2, stat_total, label='Binned Spectral Sum')
        plt.title(in_dir)
        #plt.legend()

    plt.grid()
    plt.xlabel('m/z (Da)')
    plt.ylabel('counts (A.U.)')

    plt.show()


if __name__ == '__main__':
    main()
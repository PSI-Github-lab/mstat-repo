# coding: utf-8
try:
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    from mstat.dependencies.helper_funcs import *
    from mstat.dependencies.file_conversion.QuadCConversion import quadc_to_numpy_matrix
    from mstat.dependencies.directory_dialog import *
except ModuleNotFoundError as e:
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def main():
    file_log = 'filelog'

    # define working directory
    dirhandler = DirHandler(log_name=file_log, log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    # get file from dialog
    if 'PREV_SOURCE' in dirs:
        in_dir = getFileDialog("Choose a file", "TXT files (*.txt)|*.txt", dirs['PREV_SOURCE'])
    else:
        in_dir = getFileDialog("Choose a file", "TXT files (*.txt)|*.txt")
    if len(in_dir) == 0:
        print('Action cancelled. No directories were selected.')
        quit()
    dirhandler.addDir('PREV_SOURCE', in_dir)

    dirhandler.writeDirs()

    start_time = time.time()

    mzs, intens, _, _, metadata = quadc_to_numpy_matrix(in_dir)

    print(f"--- completed in {time.time() - start_time} seconds ---")

    print('Bins Shape', mzs.shape, 'Counts Shape', intens.shape)
    print('m/z limits', metadata['lowlim'], metadata['uplim'])

    spectrum = np.sum(intens, axis=0)

    figure = plt.figure()

    plt.plot(mzs[0], spectrum, label='Spectral Sum')
    plt.plot(mzs[0], intens[-1], label='Last Scan')

    bin_size = 0.5
    low_lim = float(metadata['lowlim'])
    up_lim = float(metadata['uplim'])
    bins, num_bins = calcBins(low_lim, up_lim, bin_size)
    stats, bin_edges, _ = binned_statistic(mzs[0], spectrum, 'mean', bins=num_bins, range=(low_lim, up_lim))

    stats[np.isnan(stats)] = 0
    plt.plot(bin_edges[:-1] + bin_size/2, stats, label='Binned Spectral Sum')
    plt.title(os.path.basename(metadata['filename']))
    plt.grid()
    plt.xlabel('m/z (Da)')
    plt.ylabel('counts (A.U.)')
    plt.legend()

    figure = plt.figure()
    plt.plot(np.sum(intens, axis=1), label='TIC')
    plt.title(os.path.basename(metadata['filename']))
    plt.grid()
    plt.xlabel('Scan #')
    plt.ylabel('counts (A.U.)')
    plt.legend()
    
    plt.show()


if __name__ == '__main__':
    main()
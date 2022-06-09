from dependencies.file_conversion.RAWConversion import raw_to_numpy_array


try:
    import sys
    from subprocess import *
    import os.path
    import time
    import numpy as np
    from matplotlib import pyplot as plt, cm
    import colorcet as cc
    from dependencies.directory_dialog import *
    from scipy.stats import binned_statistic
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d
    from scipy.ndimage.morphology import white_tophat
    from dependencies.file_conversion.RAWConversion import raw_to_numpy_matrix
except ModuleNotFoundError as e:
    print(e)
    print('Install the module via "pip install ' + str(e).split("'")[-2] + '" in a CMD window and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def calcBins(low, up, bin_size, mz_lims = None):
    if mz_lims is None:
        mz_lims = [-np.inf,np.inf]
    lowlim = max(low, min(mz_lims))
    uplim = min(up, max(mz_lims))
    num_bins = int((uplim - lowlim)/bin_size)
    return np.linspace(lowlim, uplim - bin_size, num_bins), num_bins

def getAvgBinnedSpectrum(mzs, intens, binsize, low, up):
    bins, num_bins = calcBins(low, up, binsize) 

    row = np.zeros(num_bins, dtype=object)    # initialize the row which will contain the binned totals added for each scan
    # iterate through all the scans in one file
    for mz, inten in zip(mzs, intens):
        # perform binning operation
        bin_means = binned_statistic(mz, inten, 'sum', bins=num_bins, range=(low, up))[0]
        bin_means[np.isnan(bin_means)] = 0

        # add binned scan to the sum
        row = row + bin_means

    # add up all bins in the row to get a "total" row value
    total = sum(row)

    return bins, row

def getAvgSpectrum(mzs, intens):
    mean = np.zeros(mzs.shape, dtype=object)
    # iterate through all the scans in one file
    for inten in intens:
        mean = mean + np.array(inten)

        # add binned scan to the sum

    # add up all bins in the row to get a "total" row value
    total = sum(mean)

    return mzs, mean

help_message = """
Console command: python PlotRAW.py <file number> <run RAW conversion>
    NOTE: File selection done after pressing ENTER."""

def main():
    # define all directories to run the coversion
    dirhandler = DirHandler(os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()
   
    # define all directories to run the coversion
    if 'PREV_SOURCE' in dirs:
        file = getFileDialog("Choose RAW file", "RAW files (*.raw)|*.raw", dirs['PREV_SOURCE'])
        in_dir = os.path.dirname(file)
    else:
        file = getFileDialog("Choose RAW file", "RAW files (*.raw)|*.raw")
        in_dir = os.path.dirname(file)
    if len(in_dir) == 0:
        print('Action cancelled. No file selected.')
        quit()
    dirhandler.addDir('PREV_SOURCE', in_dir)
    dirhandler.writeDirs()
    
    print("Opening", os.path.basename(file))
    start_time = time.time()
    try:
        _, matrix, tics, times, _ = raw_to_numpy_matrix(file)
        rmzs, intens, _ = raw_to_numpy_array(file, sel_region=True, smoothing=False)
        mzs, corr_intens, metadata = raw_to_numpy_array(file, sel_region=True, smoothing=False)
    except Exception as exc:
        print("ERROR: no file/data found")
        print(exc)
        quit()
    #print("Raw Spect\t", rmzs.shape, intens.shape)
    #print("Corrected Spect\t", mzs.shape, corr_intens.shape)
    print(metadata)

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_title("Signal Intesity vs Time")
    ax1.plot(60.0 * times, tics, 'o-', color='black')
    ax1.set_xlabel('t (sec)')
    ax1.set_ylabel('Intesity (A.U.)')

    start = int(metadata['startscan'])
    end = int(metadata['endscan'])

    # show window of interest
    ax1.axvline(60.0 * times[start], color='green', linestyle='--') #60.0 * data['times'][start])
    ax1.axvline(60.0 * times[end], color='green', linestyle='--') #60.0 * data['times'][end])
    ax1.axvspan(60.0 * times[start], 60.0 * times[end], color='green', alpha=0.35)

    fig2, ax2 = plt.subplots(1, 1)
    # plot raw spectrum
    ax2.plot(mzs, intens / max(intens), color='black', linestyle='-.', label='Raw Spectrum')
    
    # plot filtered spectrum 
    ax2.plot(mzs, corr_intens / max(corr_intens), color='green', linestyle='--', label='Filtered Spectrum')

    # plot binned and background subtracted spectrum
    low = 100
    up = 1000 
    bin_size = 1
    bins, num_bins = calcBins(low, up, bin_size)
    stats, bins, _ = binned_statistic(mzs, corr_intens, 'sum', bins=num_bins, range=(low, up))
    corr_stats = white_tophat(stats.astype(np.float64), 15)
    ax2.plot(bins[:-1], corr_stats / max(corr_stats), color='blue', linestyle='-', marker='o', label='Binned + Bkgnd Sbtr Spectrum')

    ax2.set_title(metadata['filename'])
    ax2.set_xlabel('m/z (Dalton)')
    ax2.set_ylabel('Intesity (A.U.)')
    ax2.xaxis.set_ticks(np.arange(low, up, 50)) 
    ax2.legend()
    
    fig1.tight_layout(pad=0.25)
    fig2.tight_layout(pad=0.25)
    plt.show()

    print("STATUS: Program completed successfully!")
    print("--- completed in %s seconds ---" % (time.time() - start_time))
        

if __name__ == '__main__':
    main()
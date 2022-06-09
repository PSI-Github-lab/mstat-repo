import sys
from subprocess import *
import os.path
import time

import numpy as np
from matplotlib import pyplot as plt, cm
import colorcet as cc

from dependencies.file_conversion.RAWConversion import *
from dependencies.directory_dialog import *
import wx
import wx.lib.agw.multidirdialog as MDD

from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.ndimage.morphology import white_tophat

def getDirFromDialog():
    # create file dialog using wxApp
    app = wx.App(0)
    dlg = MDD.MultiDirDialog(None, title="Choose folders containing raw files for conversion", defaultPath="C:/Users/Jackson/PSI Files Dropbox/MS Detectors/LTQ/",
                            agwStyle=MDD.DD_MULTIPLE|MDD.DD_DIR_MUST_EXIST)

    if dlg.ShowModal() != wx.ID_OK:
        print("You Cancelled The Dialog!")
        dlg.Destroy()
        return []

    else:
        paths = dlg.GetPaths()

        directories = [
            path[1].replace('Local Disk (C:)', 'C:')
            for path in enumerate(paths)
        ]

    dlg.Destroy()
    app.MainLoop()

    return directories

def getAvgWindow(tics):
    avg = np.mean(tics)
    print(avg)
    start = next(x[0] for x in enumerate(tics) if x[1] >= avg)
    reverse = tics[::-1]
    end = -next(x[0] for x in enumerate(reverse) if x[1] >= avg) - 1
    print(start, end)
    return start, end

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
    mean = np.zeros(len(mzs[0]), dtype=object)
    # iterate through all the scans in one file
    for inten in intens:
        mean = mean + np.array(inten)

        # add binned scan to the sum

    # add up all bins in the row to get a "total" row value
    total = sum(mean)

    return mzs[0], mean

def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    From https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data 
    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed

def handleStartUpCommands(help_message):
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

help_message = """
Console command: python PlotRAW.py <file number> <run RAW conversion>
    NOTE: Directory selection done after pressing ENTER.
Arguments:
    <file number>        - (Integer) which file in the selected directory you want to plot.
    <run RAW conversion> - (Boolean) turn RAW file conversion on or off."""

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)

    # define working directory
    if len(argm) == 0:
        print("ERROR: Please provide arguments when calling the script. Type 'python rawToCSV.py help' for more information.")
        quit()
   
    # define all directories to run the coversion
    dirhandler = DirHandler(os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()
    #print(dirs)
   
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

    # define user passed arguments
    file_num = int(argm[0])

    mzmlDirectories, classes = rawToMZML(in_directories, int(argm[1]))
    
    data = mzmlDirectories[0].grabFileData(file_num)
    if type(data) is int:
        print("ERROR: no MZML file/data found")
        quit()
    #print(data['times'].shape, data['mzs'].shape, data['intens'].shape)

    fig, axes = plt.subplots(3, 1)
    axes[0].set_title("Signal Intesity vs Time")
    axes[0].plot(60.0 * data['times'], data['tics'], 'o-', color='black')
    axes[0].set_xlabel('t (sec)')
    axes[0].set_ylabel('Intesity (A.U.)')

    # show window of interest
    start, end = getAvgWindow(data['tics'])
    axes[0].axvline(60.0 * data['times'][start], color='green', linestyle='--') #60.0 * data['times'][start])
    axes[0].axvline(60.0 * data['times'][end], color='green', linestyle='--') #60.0 * data['times'][end])
    axes[0].axvspan(60.0 * data['times'][start], 60.0 * data['times'][end], color='green', alpha=0.35)

    #get average spectrum in window of interest
    low = 100 #np.floor(np.min([min(x) for x in data['mzs'][start:end] if len(x) > 0]))
    up = 1000 #np.ceil(np.max([max(x) for x in data['mzs'][start:end] if len(x) > 0]))
    print(data['mzs'][start:end].shape, data['intens'][start:end].shape)
    mzs, spect = getAvgSpectrum(data['mzs'][start:end], data['intens'][start:end])

    
    # apply filter before binning
    window = 51
    order = 4
    t = 5
    # perform some signal pre-processing
    #mvgavg_spect = uniform_filter1d(spect.astype(np.float64), 5)
    savgol_spect = savgol_filter(spect, window, order)
    corr_spect = savgol_spect#white_tophat(savgol_spect.astype(np.float64), t)
    """
    filt_scans = []
    for mz, scan in zip(data['mzs'][start:end], data['intens'][start:end]):
        if len(mz) > window:
            filt_scans.append(non_uniform_savgol(mz, scan, window, order))
        else:
            filt_scans.append(scan)
    
    filt_bins, filt_spect = getAvgSpectrum(data['mzs'][start:end], filt_scans, 1, low, up)
    """

    i = np.argmax(data['tics'])
    axes[1].set_title("Selected Spectrum")
    axes[1].plot(mzs, spect, color='black')
    #axes[1].plot(mzs, corr_spect, color='green')
    axes[1].set_xlabel('m/z (Dalton)')
    axes[1].xaxis.set_ticks(np.arange(low, up, 50)) 
    axes[1].set_ylabel('Intesity (A.U.)')

    axes[2].set_title("Pre-processed Spectra")
    axes[2].plot(mzs, spect, color='black')
    #axes[2].plot(bins, mvgavg_spect, color='red')
    #axes[2].plot(bins, white_tophat(spect.astype(np.float64), t))
    #axes[2].plot(bins, savgol_spect, color='blue')
    bin_size = 1
    bins, num_bins = calcBins(low, up, bin_size)
    axes[2].plot(mzs, corr_spect, color='green', label='Filt + Bckgnd Subtract')
    stats, bins, _ = binned_statistic(mzs, corr_spect, 'sum', bins=num_bins, range=(low, up))
    corr_stats = white_tophat(stats.astype(np.float64), 5)
    axes[2].plot(bins[:-1], corr_stats, color='blue', label='Binned')
    axes[2].set_xlabel('m/z (Dalton)')
    axes[2].xaxis.set_ticks(np.arange(low, up, 50)) 
    axes[2].set_ylabel('Intesity (A.U.)')
    axes[2].legend()
    
    fig.tight_layout(pad=0.25)
    plt.show()

    print("STATUS: Program completed successfully!")
        

if __name__ == '__main__':
    main()
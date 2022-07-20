# coding: utf-8
try:
    import numpy as np
    from matplotlib import pyplot as plt
    import configparser as cp
    from logging import config
    from os.path import exists
    import sys
    import time
    import os
    from scipy import signal
    from pymsfilereader import MSFileReader
    from scipy.ndimage.morphology import white_tophat
    from sklearn.model_selection import train_test_split, LeaveOneOut
    from scipy.stats import binned_statistic
    from tqdm import tqdm
    import glob
    from mstat.dependencies.file_conversion.RAWConversion import raw_to_numpy_array
    from mstat.dependencies.directory_dialog import *
    from mstat.dependencies.helper_funcs import getTICNormalization

except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class ConfigHandler:
    new_config = False
    config_created = False

    def __init__(self, config_name="config.ini", my_path = ".") -> None:
        self.config = cp.RawConfigParser()
        self.config_name = config_name
        self.my_path = my_path

    def read_config(self) -> bool:
        if exists(self.config_name):
            self.config.read(self.config_name)
            self.config_created = True
            return True
        return False

    def write_config(self) -> bool:
        if self.config_created:
            with open(self.config_name, 'w') as config_file:
                self.config.write(config_file)
            return True
        return False

    def set_option(self, section_name : str, option_name : str, value) -> None:
        if type(value) is str:
            self.config.set(section_name, option_name, value)
        else:
            self.config.set(section_name, option_name, str(value))

    def get_option(self, section_name : str, option_name : str, value='no val') -> str:
        return self.config.get(section_name, option_name, fallback=value)

    def create_config(self, section_names : list) -> None:
        for name in section_names:
            self.config.add_section(name)

        self.config_created = True
        self.new_config = True   

def process_single_file(file_name, start, end):
    try:
        raw_file = MSFileReader(file_name)
    except OSError as exc:
        raise exc
    num_scans = raw_file.GetLastSpectrumNumber()

    if num_scans < end:
        end = num_scans

    data = raw_file.GetSummedMassSpectrum([1])[0]    # this function does most of the work
    mzs = np.array(data[0])
    #print(num_scans, len(data[1]), len(mzs))
    intens = np.zeros((num_scans, len(data[1])))
    #print(intens[0, :].shape, np.array(data[1]).shape)
    intens[0, :] = np.array(data[1])
    for i in tqdm(range(2, num_scans+1), total=num_scans, desc='\nGetting scans from file'):
        data = raw_file.GetSummedMassSpectrum([i])[0]
        if mzs.shape[0] != len(data[0]):
            raise ValueError("Scans do not all share the same number of bins.")
        new_intens = np.array(data[1])
        new_intens[np.isnan(new_intens)] = 0
        intens[i-1,:] = new_intens

    scans = intens[start:end, :]
    tics = np.sum(scans, axis=1)

    norm_data = np.empty(0)
    for row, total in tqdm(zip(scans, tics), total=scans.shape[0], desc=f'Combining Scans {start}-{end}'):
            print(row, total)
            try:
                norm_data = np.vstack((norm_data, row / total))
            except ValueError:
                norm_data = row / total
    print('')
    return norm_data, start, end

def process_directory(path : str, start : int, end : int, scan_sel : bool, filt : bool):
    num_raw = len(glob.glob1(path,"*.raw"))
    print(f"\n{num_raw} RAW files found in {path}")

    if num_raw < end:
        end = num_raw
    if num_raw == 0:
        quit()

    num_scans = []
    norm_data = np.empty(0)
    file_names = []
    tics = []
    lims = (0.0, 0.0)
    start_time = time.time()
    for file in tqdm(os.listdir(path), total=num_raw, desc=f'Combining File Spectra {start}-{end}'):
        if file.endswith('raw'):
            mzs, intens, metadata = raw_to_numpy_array(rf'{path}\{file}', sel_region=scan_sel, smoothing=filt)
            #print("Start & End Scan", (metadata['startscan'],metadata['endscan'], int(metadata['endscan'])-int(metadata['startscan'])))
            num_scans.append(int(metadata['numscans']))
            #stats, bins, _ = binned_statistic(mzs, intens, 'sum', bins=900, range=(100, 1000))
            intens[np.isnan(intens)] = 0.
            total = sum(intens)
            tics.append(total)
            file_names.append(file)
            #print(total)
            if total == 0.0:
                print("ZERO DATA", intens.max(), file)
            else:
                if lims not in [(float(metadata['lowlim']), float(metadata['uplim'])), (.0, .0)]:
                    raise ValueError(f"One or more files do not match the {lims} m/z limits")
                try:
                    norm_data = np.vstack((norm_data, intens / total))
                except ValueError as exc:
                    norm_data = intens / total
                    lims = (float(metadata['lowlim']), float(metadata['uplim']))
    print('')
    print("completed in %s seconds".center(80, '*') % (time.time() - start_time))

    return norm_data, np.array(tics), file_names, num_scans, start, end

from audioop import cross
from random import random
from scipy import rand
from scipy.stats import binned_statistic
def calcBins(low, up, bin_size, mz_lims = None):
    if mz_lims is None:
        mz_lims = [-np.inf,np.inf]
    lowlim = max(low, min(mz_lims))
    uplim = min(up, max(mz_lims))
    num_bins = int((uplim - lowlim)/bin_size)
    return np.linspace(lowlim, uplim-bin_size, num_bins), num_bins

def bin_data(mzs, intens, meta, bin_size=1):
    #print('BINNING', mzs.shape, intens.shape)
    low = float(meta[0]['lowlim'])
    up = float(meta[0]['uplim'])
    #print(low, up, bin_size)
    bins, num_bins = calcBins(low, up, bin_size)
    rows = []

    for intens_row, mzs_row in zip(intens, mzs):
        stats, bin_edges, _ = binned_statistic(mzs_row, intens_row, 'mean', bins=num_bins, range=(low, up))
        #print(stats[:5], sum(row), sum(stats))
        new_row = np.concatenate(([sum(stats)], stats)) 
        rows.append(stats)

    return bin_edges + bin_size/2, np.array(rows)

def calc_spectral_SNR(norm_data, randomize, start, end, step):
    m = norm_data.shape[0]
    n = norm_data.shape[1]
    #print((m, n), type(norm_data))

    # show some spectra
    """
    numb = 100
    fig, ax = plt.subplots(1,1)
    minspectrum, maxspectrum = np.min(norm_data), 1.1*np.max(norm_data)
    for i, spectrum in enumerate(norm_data[:5]):
        ax[0].plot(spectrum + i*0.0*maxspectrum, lw=0.5, label=f'Scan {i+1}')
    #ax[0].plot(t, signal_clean, color='r', lw=1, label='Clean Signal')
    #ax[0].set_ylim([minspectrum, maxspectrum])

    fig.suptitle(os.path.basename(path))

    avg_data = np.sum(norm_data, axis=0) / norm_data.shape[0]
    ax[0].plot(avg_data, lw=0.5, label='Average Spectrum')

    ax[0].set_xlabel('bins')
    ax[0].set_ylabel('intensity')
    ax[0].legend()"""
    if randomize:
        #print('RANDOMIZE')
        #np.random.seed(2)
        np.random.shuffle(norm_data)

    SNR = []
    x = []
    first = max(step, 2)
    final = end
    for j in range(first, int(final/step)*step+1, step): #tqdm(, total=int(final/step), desc="Creating SNR Plot"):
        super_spectrum = np.empty((j*n,))
        for i, spectrum in enumerate(norm_data[:j]):
            super_spectrum[i*n:(i+1)*n] = spectrum

        #print(super_spectrum.shape)

        f, Pxx = signal.periodogram(super_spectrum, n)
        #f2, Pxx2 = signal.welch(super_spectrum, n)
        #Pxx, f = mlab.psd(super_spectrum, n)
        '''
        l = j*n
        dt = 1/n
        fhat = np.fft.fft(super_spectrum) #computes the fft
        psd = fhat * np.conj(fhat) / (l*n)  # factor of 2 to match periodogram... why?
        freq = (1/m) * np.arange(l) #frequency array
        idxs_half = np.arange(0, np.floor(l/2), dtype=np.int32) #first half index
        psd_real = psd[idxs_half] #amplitude for first half
        psd_real[0] = 0.0       # don't care about DC
        freq = freq[idxs_half]'''


        df = f[1]-f[0]
        #print(f"df: {df}", np.where(f == 1.0)[0][0])

        S = 0.
        num_s = 0
        N = 0.
        num_n = 0

        for k, (ff, pp) in enumerate(zip(f, Pxx)):
            if k % j == 0:#np.mod(ff, 1.0) < 0.5*df:
                S += pp
                num_s += 1
            else:
                N += pp
                num_n += 1

        #ax[1].semilogy(freq, psd_real / max(psd_real), color='b', lw=0.5, label='FFT PSD')
        #ax[1].semilogy(f, Pxx / max(Pxx), color='r', lw=0.5, label='Spectrogram PSD')
        #ax[1].semilogy(f2, Pxx2, color='g', lw=0.5, label='Welch PSD')
        snr = np.sqrt((S / num_s) / (N / num_n))
        #if num_scans is not None:
        #    snr = snr #/ np.sqrt(np.mean(num_scans[:j]))
        SNR.append(snr)
        x.append(j)
        #print(f"\nSignal ({S}, {num_s}), Noise ({N}, {num_n})")
        #print(f"SNR {np.sqrt((S / num_s) / (N / num_n))}\n")

    return SNR, x

def estimate_SNR_series(path, start, end, step=5, scan_sel=False, filt=False, crossval='none', randomize=False):
    # read data from file(s)
    start_time = time.time()
    num_scans = None
    if '.raw' in path.lower():
        # only single file has been passed
        # look at scans in the single file
        norm_data, start, end = process_single_file(path, start, end, scan_sel, filt)
    else:
        # directory has been passed
        # check for raw files
        if len(glob.glob1(path,"*.npy")) > 0:
            with open(rf'{path}\{os.path.basename(path)}.npy', 'rb') as f:
                intens = np.load(f, allow_pickle=True)
                mzs = np.load(f, allow_pickle=True)
                meta = np.load(f, allow_pickle=True)

                label = meta[0]['comment1']
                if label == '':
                    label = os.path.basename(path)

                print(label)
                print(intens.shape)
                _, binned_intens = bin_data(mzs, intens, meta)
                print(binned_intens.shape)
                
                norm_data = getTICNormalization(binned_intens)
                tics = np.sum(intens, axis=1)
                file_names = [entry['filename'] for entry in meta]

                end=intens.shape[0]
        else:
            norm_data, tics, file_names, num_scans, start, end = process_directory(path, start, end, scan_sel, filt)

    
        
    if crossval == 'loo':
        print(f"Performing leave-one-out (loo) cross validation on {norm_data.shape[0]} samples.\nMaximum number of SNR points is {norm_data.shape[0]-1}")
        randomize = True
        if randomize:
        #print('RANDOMIZE')
        #np.random.seed(2)
            np.random.shuffle(norm_data)
        loo = LeaveOneOut()
        n_splits = loo.get_n_splits(norm_data)

        SNRs = []
        xs = []
        for train_index, test_index in tqdm(loo.split(norm_data), total=n_splits, desc='LOO S/N ESTIMATION'):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = norm_data[train_index], norm_data[test_index]
            SNR, x = calc_spectral_SNR(X_train, randomize, start, end-1, step)
            SNRs.append(np.array(SNR))
            xs.append(np.array(x))

        SNRs = np.array(SNRs)
        xs = np.array(xs)

        snr_mean = np.mean(SNRs, axis=0)
        snr_std = np.std(SNRs, axis=0, ddof=1)
        
    elif '-' in crossval.lower():
        fraction = 1 - float(crossval.split('-')[0])/100
        reps = int(crossval.split('-')[1])
        print(f"Performing cross validation on {100*fraction}% of samples with {reps} repetions.\nMaximum number of SNR points is {int(norm_data.shape[0]*fraction)}")
    
        SNRs = []
        xs = []
        for _ in tqdm(range(reps), total=reps, desc=f'{crossval} S/N ESTIMATION'):
            X_train, _, = train_test_split(norm_data, test_size=1-fraction)
            SNR, x = calc_spectral_SNR(X_train, randomize, start, end-int(norm_data.shape[0]*(1-fraction)), step)
            SNRs.append(np.array(SNR))
            xs.append(np.array(x))

        SNRs = np.array(SNRs)
        xs = np.array(xs)

        snr_mean = np.mean(SNRs, axis=0)
        snr_std = np.std(SNRs, axis=0, ddof=1)

    else:
        print(f"No cross validation to be preformed. Using single {'random' if randomize else 'directory'} order.")
        SNR, x = calc_spectral_SNR(X_train, randomize, start, end, step)

        snr_mean = np.array(SNR)
        snr_std = None

        SNR, x = calc_spectral_SNR(norm_data, randomize, start, end, step)

    print("\nNumber of Scans:", num_scans)
    print("--- completed in %s seconds ---" % (time.time() - start_time))
    return file_names, tics, snr_mean, snr_std, x

help_message = """
Console Command: python SNREstimate.py <path/base_file_name.csv>
Arguments:
    <path/file(s)>    - (String) path and name of RAW file including the extension ".raw" or directory containing RAW files"""

def getAvgSpectrum(mzs, intens):
    mean = np.zeros(len(mzs), dtype=object)
    # iterate through all the scans in one file
    i = 0
    for inten in intens:
        mean = mean + np.array(inten)
        i += 1

        # add binned scan to the sum

    # add up all bins in the row to get a "total" row value
    total = sum(mean)

    return mzs[0], mean / i

def handleStartUpCommands(help_message):
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
    config_file = 'snrest_config.ini'
    config_hdlr = ConfigHandler(config_name=config_file)
    if not config_hdlr.read_config():
        config_hdlr.create_config(['SETTINGS'])
        config_hdlr.set_option('SETTINGS', 'scansel(y/n)',  'n')
        config_hdlr.set_option('SETTINGS', 'filter (y/n)',  'n')
        config_hdlr.set_option('SETTINGS', 'crossval(loo/{percent left out}-{num repetitions}/none)', 'loo')
        config_hdlr.set_option('SETTINGS', 'randord(y/n)',  'y')
        config_hdlr.set_option('SETTINGS', 'plottic(y/n)',  'n')
        config_hdlr.set_option('SETTINGS', 'xaxstep(int)',  5)

        config_hdlr.write_config()
    
    scan_sel = ('y' == config_hdlr.get_option('SETTINGS', 'scansel(y/n)', 'n').lower())
    filt = ('y' == config_hdlr.get_option('SETTINGS', 'filter (y/n)', 'n').lower())
    crossval = config_hdlr.get_option('SETTINGS', 'crossval(loo/{percent left out}-{num repetitions}/none)', 'loo').lower()
    rand = ('y' == config_hdlr.get_option('SETTINGS', 'randord(y/n)', 'n').lower())
    do_tic_plot = ('y' == config_hdlr.get_option('SETTINGS', 'plottic(y/n)', 'n').lower())
    xax_step = int(config_hdlr.get_option('SETTINGS', 'xaxstep(int)', 5))

    print(f"CHANGE SETTINGS IN '{config_file}'".center(80, '*'))
    print("\tCurrent settings", scan_sel, filt, crossval, rand, xax_step)

    fig1, ax = plt.subplots(1,1)
    '''if do_tic_plot:
        fig2, ax2 = plt.subplots(1,1)
        ax2.set_xlabel('sample (in alphabetical order)')
        ax2.set_ylabel('TIC (A.U.)')
        ax2.grid()'''

    # get directories
    dirhandler = DirHandler(log_name='snrest', log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    # define all directories to run the coversion
    if 'PREV_SOURCE' in dirs:
        in_directories = getMultDirFromDialog("Choose folders containing RAW files for analysis", dirs['PREV_SOURCE'])
    else:
        in_directories = getMultDirFromDialog("Choose folders containing RAW files for analysis")
    if len(in_directories) == 0:
        print('Action cancelled. No directories were selected.')
        quit()
    common_src = os.path.commonpath(in_directories)
    dirhandler.addDir('PREV_SOURCE', common_src)

    dirhandler.writeDirs()

    start = 0
    end = 10000
    max_plot_value = .0
    for path in in_directories:
        file_names, tics, snr_mean, snr_std, x = estimate_SNR_series(path, start, end, step=xax_step, scan_sel=scan_sel, filt=filt, crossval=crossval, randomize=rand)
        dirpath, dir = os.path.split(path)
        dirpath, parent_dir = os.path.split(dirpath)
        if do_tic_plot:
            plt.figure()
            plt.xlabel('sample (in alphabetical order)')
            plt.xticks(rotation = 90)
            plt.ylabel('TIC (A.U.)')
            plt.grid()
            plt.plot(file_names, tics, label=f'{parent_dir}\{dir}', marker='o')
            plt.legend()
        print(f"Directory: \t\t{parent_dir}\{dir}")
        print(f"SNR:\t\t{snr_mean}")
        print(f"# samples:\t{x}")
        ax.plot(x, snr_mean, label=f'{parent_dir}\{dir}', marker='o')
        ax.fill_between(x, snr_mean - 2*snr_std,
                         snr_mean + 2*snr_std, alpha=0.1,
                         #color="g"
                         )
        max_plot_value = max(max_plot_value, max(snr_mean + 2*snr_std))
    ax.set_ylim([0, max_plot_value])
    ax.set_xlabel('# samples')
    ax.set_ylabel('SNR')
    ax.grid()
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
import sys
from datetime import *
import time

from mstat.dependencies.ms_data.MSFileReader import MSFileReader

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

help_message = """
Console Command: python DenoiseSpectra.py <path/base_file_name.csv>
Arguments:
    <path/base_file_name.csv>    - (String) path and name of base data CSV file including the extension ".csv" """

def handleStartUpCommands(help_message):
    argm = [arg for arg in sys.argv[1:]]
    if len(argm) != 0 and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

def main():
    argm = handleStartUpCommands(help_message)
    if not argm:
        print("Type 'python DenoiseSpectra.py help' for more info")
        quit()
    else:
        file_name = argm[0]
        rnd_state = 44

    # read training data from the csv file
    if file_name == 'rand':
        class_spectra = np.array([
            #[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],
            #[1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1],
            #[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
            [0,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,3,3,3,3,3,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,3,3,3,3,0,0,0]
            #[1,2,3,4]
        ])
        class_spectra = np.ones((64,1))
        m = class_spectra.shape[0]
        n = class_spectra.shape[1]
        DFT = np.empty((m*n, m*n), dtype=np.complex_)
        for i in range(DFT.shape[0]):
            for j in range(DFT.shape[1]):
                DFT[i, j] = np.exp((-1j * 2 * np.pi * i * j) / (m*n))

        cm = plt.get_cmap('twilight')
        fig, ax = plt.subplots(1,1)
        plt.imshow(-np.angle(DFT) % (2*np.pi), cmap=cm)
        plt.clim(0, 2*np.pi)

        #print(np.mean(avg_spectrum), fhat_clean[0]/len)
    else:
        file_reader = MSFileReader(file_name)
        _, feature_data, training_labels, encoder = file_reader.encodeData()
        print(file_reader)

        # get data from a single class
        spectral_data = file_reader.getTICNormalization()
        classes = encoder.classes_
        class_choice = classes[0]
        print(class_choice)
        class_spectra = spectral_data[(training_labels == class_choice)]
    m = class_spectra.shape[0]
    n = class_spectra.shape[1]
    print((m, n))

    #spectrum = class_data[0, :]
    fig, ax = plt.subplots(3,1)

    # show some spectra
    numb = m
    minspectrum, maxspectrum = np.min(class_spectra), 1.1*np.max(class_spectra)
    for i, spectrum in enumerate(class_spectra[:5, :]):
        ax[0].plot(spectrum + i*0.0*maxspectrum, lw=0.5, label=f'Noisy Spectrum {i}')
    #ax[0].plot(t, signal_clean, color='r', lw=1, label='Clean Signal')
    ax[0].set_ylim([minspectrum, maxspectrum])
    ax[0].set_xlabel('Bins')
    ax[0].set_ylabel('Intensity')
    ax[0].legend()

    with Timer('AVG'):
        avg_spectrum = np.mean(class_spectra, axis=0)

    ## Compute Fourier Transform
    with Timer('FFT'):
        super_spectrum = np.empty((numb*n,))
        for i, spectrum in enumerate(class_spectra[:numb, :]):
            super_spectrum[i*n:(i+1)*n] = spectrum
        len = m*n
        dt = 1
        fhat = np.fft.fft(super_spectrum, len) #computes the fft
        
        freq = (1/(dt*len)) * np.arange(len) #frequency array
        idxs_half = np.arange(0, np.floor(len/2), dtype=np.int32) #first half index
        fhat_real = np.abs(fhat) #amplitude for first half [idxs_half]

        ## Filter out noise
        idxs_filt = np.array([1 if (i % m == 0) else 0 for i in range(len)]) #and (i > 20*m) and (i < len-20*m)   (i < len//2) and ((i < 200*m) or (i > len-200*m))
        print(fhat[2], fhat[-2])
        fhat_clean = fhat * idxs_filt #used to retrieve the signal
        fhat_clean_real = np.abs(fhat_clean)

        spectra_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform
    
    ax[1].plot(fhat_real, color='b', lw=0.5, label=r'$|\mathcal{F}_{noisy}|$') #freq[idxs_half], 
    ax[1].plot(fhat_clean_real, color='r', lw=0.5, label=r'$|\mathcal{F}_{clean}|$')
    ax[1].set_xlabel('Frequencies')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()

    # Plot filtered spectrum
    ax[2].plot(np.abs(spectra_filtered[:n]), color='b', lw=1, label='Clean Spectrum')

    # Plot average spectrum for comparison
    ax[2].plot(avg_spectrum[:n], color='r', lw=1, label='Average Spectrum')
    minspectrum, maxspectrum = np.min(np.abs(spectra_filtered[:n])), 1.1*np.max(np.abs(spectra_filtered[:n]))
    ax[2].set_ylim([minspectrum, maxspectrum])
    ax[2].set_xlabel('Bins')
    ax[2].set_ylabel('Intensity')
    ax[2].legend()

    plt.show()

if __name__ == "__main__":
    main()
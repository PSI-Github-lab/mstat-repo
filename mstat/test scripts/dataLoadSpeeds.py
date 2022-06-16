import numpy as np
import os
import time
from dependencies.file_conversion.MZMLDirectory import MZMLDirectory
from scipy.stats import binned_statistic
from matplotlib import mlab, pyplot as plt, cm

def calcBins(low, up, bin_size, mz_lims = None):
        if mz_lims is None:
            mz_lims = [-np.inf,np.inf]
        lowlim = max(low, min(mz_lims))
        uplim = min(up, max(mz_lims))
        num_bins = int((uplim - lowlim)/bin_size)
        return np.linspace(lowlim, uplim-bin_size, num_bins), num_bins

bin_size = 1.0
low = 100
up = 1000


#print("Getting binned data from MZML file")
#start_time = time.time()
#mzml_dir = MZMLDirectory(r'C:\Users\Jackson\PSI Files Dropbox\MS Detectors\LTQ\LTQ Velos Data\Handheld_IonTrapOnly\AtlanticSalmon\2022-01-20_MoreBurning_MW\MZML')
#data, bins, files = mzml_dir.createArray()
#print(f"--- Ran in {time.time() - start_time} seconds ---") 

from pymsfilereader import MSFileReader

print("Getting data from RAW files")
path = r'C:\Users\Jackson\PSI Files Dropbox\MS Detectors\LTQ\Testing Data\AtlanticSalmon\2022-01-20_MoreBurning_MW'

start_time = time.time()

# bypassing mzml creation by dumping data from raw to numpy files
raw_files = []
mzs = []
intens = []
metadata = []
for file in os.listdir(path):
        if file.endswith('raw'):
            raw_file = MSFileReader(rf'{path}\{file}')
            raw_files.append(raw_file)
            num_scans = raw_file.GetLastSpectrumNumber()
            data = raw_file.GetSummedMassSpectrum(list(range(1,num_scans+1)))[0]    # this function does most of the work
            mzs.append(np.array(data[0]))
            intens.append(np.array(data[1]))
            metadata.append([str(file), str(raw_file.GetComment1()), str(raw_file.GetComment2()), str(raw_file.GetLowMass()), str(raw_file.GetHighMass())])

mzs = np.array(mzs)
intens = np.array(intens)
metadata = np.array(metadata)

print(f"--- Ran in {time.time() - start_time} seconds ---") 

#print(mzs[:10, :5])

print(type(intens), type(mzs[0]), type(metadata))
print(intens.shape, mzs[0].shape, metadata.shape)

file_name = 'test.npy'
#with open(file_name, 'wb') as f:
#    np.save(f, intens)
#    np.save(f, mzs[0])
#    np.save(f, metadata)


print("Getting binned data from NPY file")
#start_time = time.time()
with open(file_name, 'rb') as f:
    intens = np.load(f)
    mzs = np.load(f)
    meta = np.load(f)

low = float(meta[0][3])
up = float(meta[0][4])
bins, num_bins = calcBins(low, up, bin_size)
rows = []

start_time = time.time()
for row in intens:
    stats, _, _ = binned_statistic(mzs, row, 'sum', bins=num_bins, range=(low, up))
    new_row = np.concatenate(([sum(stats)], stats)) 
    rows.append(new_row)
    
binned_intens, bins, meta = np.array(rows), bins + bin_size/2, meta

print(f"--- Ran in {time.time() - start_time} seconds ---")

print(binned_intens.shape, bins.shape, meta.shape)
print(bins[:10])


fig, ax = plt.subplots(1,1)
row = intens[0]
ax.plot(mzs, row, lw=0.5, label='Scan 0')
start_time = time.time()
stats, _, _ = binned_statistic(mzs.astype(np.float64), row.astype(np.float64), 'mean', bins=num_bins, range=(low, up))
print(f"--- Ran in {time.time() - start_time} seconds ---") 
#ax.plot(bins, stats, lw=0.5, label='binned 1')

ax.set_xlabel('m/z')
ax.set_ylabel('Counts')
ax.legend()

plt.show()

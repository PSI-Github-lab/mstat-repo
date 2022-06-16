try:
    from scipy.stats import binned_statistic
    from scipy.signal import savgol_filter
    from scipy.ndimage.morphology import white_tophat
    import csv
    from pyteomics import mzml
    import numpy as np
    import mmap
    import re
    import os
    import pandas as pd
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

class MZMLDirectory:
    directory: str
    mz_lims: list       # m/z range limits (max and min value)
    txt_count: int      # count number of txt files found after raw conversion
    bin_size: int
    lowlim: float
    uplim: float
    num_bins: int
    bins: list

    def __init__(self, directory) -> None:
        self.directory = directory

    def calcBins(self, low, up, bin_size, mz_lims = None):
        if mz_lims is None:
            mz_lims = [-np.inf,np.inf]
        self.lowlim = max(low, min(mz_lims))
        self.uplim = min(up, max(mz_lims))
        num_bins = int((self.uplim - self.lowlim)/bin_size)
        return np.linspace(self.lowlim, self.uplim-bin_size, num_bins), num_bins

    def checkMZLims(self):
        self.mz_lims = []
        self.txt_count = 0
        # find the range of m/z in the mzML data (from the metadata)
        for file in os.listdir(self.directory):
            # check metadata to see if everything has consistent dimensions/ranges
            if file.endswith('txt'):
                with open(self.directory + r"\\" + file) as f:
                    lines = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)   # read the lines from the file, converted into binary string
                    mr_index = lines.find(b'Mass range')    # find mass range entry in metadata by searching for the 'Mass range' field
                    if mr_index != -1:
                        fg_index = lines.find(b'Fragmentation') # find end of the mass range values by looking for the 'Fragmentation' line immediately after
                        # extract the two m/z limit numbers from the spot found in the metadata file using regex black magic
                        self.mz_lims += [
                            int(c)
                            for c in re.split(
                                '[^a-zA-Z0-9]',
                                lines[mr_index:fg_index].decode('ascii'),
                            )
                            if c.isdigit()
                        ]

                self.mz_lims = list(set(self.mz_lims))    # record only unique limit values (there should only be 2 of them)

                self.txt_count += 1
            else:
                continue
        return self.mz_lims      

    def createCSV(self, csv_name, bin_size, low, up) -> int:
        # calculate number of bins to be used when summing the scans in each file
        self.bin_size = bin_size
        self.bins, self.num_bins = self.calcBins(low, up, bin_size) 

        # create the CSV file and write summed scans as rows in the file
        with open(csv_name + '.csv', mode='w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(np.concatenate((['filename', 'total'], [str(e) for e in self.bins])))
            mzml_count = 0
            # iterate through all the mzML files
            for file_ in os.listdir(self.directory):
                if file_.endswith('mzML'):
                    with mzml.read(self.directory + r'\\' + file_, use_index=True) as reader:
                        row = np.zeros(self.num_bins, dtype=np.object)    # initialize the row which will contain the binned totals added for each scan

                        # iterate through all the scans in one file
                        for scan in reader:
                            # get m/z values and intensity values from the mzML structure
                            mz = scan['m/z array']
                            inten = scan['intensity array']

                            #frame = pd.DataFrame([mz, inten])
                            #print(frame)

                            # perform binning operation
                            bin_means = binned_statistic(mz, inten, 'sum', bins=self.num_bins, range=(low, up))[0]
                            bin_means[np.isnan(bin_means)] = 0

                            # add binned scan to the sum
                            row = row + bin_means

                        # add up all bins in the row to get a "total" row value
                        total = sum(row)
                        row = np.concatenate(([file_, total], row))    # add total and file name to the beginning of the row
                        writer.writerow(row)    # write the row to the CSV file
                    mzml_count += 1
            if mzml_count == 0:
                return 1
            else:
                return 0
    
    def createTXT(self, output_dir, txt_name) -> int:
        # create the txt files and write summed scans as rows in the file
        mzml_count = 0
        # iterate through all the mzML files
        for file_ in os.listdir(self.directory):
            if file_.endswith('mzML'):
                with mzml.read(self.directory + r'\\' + file_, use_index=True) as reader:
                    with open(output_dir + r'\\' + txt_name + '_' + file_.rsplit('.', 1)[0] + '.txt', mode='w', newline='') as output_file:
            
                        # iterate through all the scans in one file
                        for scan in reader:
                            # get m/z values and intensity values from the mzML structure
                            ind = scan['index']
                            mz = scan['m/z array']
                            inten = scan['intensity array']

                            output_file.write(str(ind) + '\t' + '\t'.join([str(e) for e in mz]) + '\n')
                            output_file.write(str(ind) + '\t' + '\t'.join([str(e) for e in inten]) + '\n')

                mzml_count = mzml_count + 1
        if mzml_count == 0:
            return 1
        else:
            return 0

    def createFrame(self, bin_size, low, up) -> int:
        # calculate number of bins to be used when summing the scans in each file
        self.bin_size = bin_size
        _, self.num_bins = self.calcBins(low, up, bin_size)

        # iterate through all the mzML files
        mzml_count = 0
        rows = []
        for file_ in os.listdir(self.directory):
            if file_.endswith('mzML'):
                with mzml.read(rf'{self.directory}\{file_}', use_index=True) as reader:
                    mzs = reader[0]['m/z array']
                    mean = np.zeros(mzs.shape[0], dtype=object)
                    # iterate through all the scans in one file
                    for scan in reader:
                        mean = mean + np.array(scan['intensity array'])
                   
                    # perform some signal pre-processing
                    # apply filter before binning
                    window = 51
                    order = 2
                    t = 50
                    savgol_spect = savgol_filter(mean, window, order)
                    corr_spect = white_tophat(mean.astype(np.float64), t)

                    # bin the pre-processed spectrum
                    stats, self.bins, _ = binned_statistic(mzs, corr_spect, 'mean', bins=self.num_bins, range=(low, up))
                    stats[np.isnan(stats)] = 0

                    # add up all bins in the row to get a "total" row value
                    total = sum(stats)
                    row = np.concatenate(([file_, total], stats))    # add total and file name to the beginning of the row
                    rows.append(row)    # write the row to the data for frame
                mzml_count += 1
        
        return pd.DataFrame(data=rows, columns=np.concatenate((['filename', 'total'], [str(e) for e in (self.bins[:-1] + self.bin_size/2)])))

    def createArray(self, bin_size=None, low=None, up=None) -> np.ndarray:
        # calculate number of bins to be used when summing the scans in each file
        self.bin_size = bin_size
        if bin_size is not None:
            _, self.num_bins = self.calcBins(low, up, bin_size)

        # iterate through all the mzML files
        mzml_count = 0
        rows = []
        file_names = []
        for file_ in os.listdir(self.directory):
            if file_.endswith('mzML'):
                with mzml.read(rf'{self.directory}\{file_}', use_index=True) as reader:
                    mzs = reader[0]['m/z array']
                    mean = np.zeros(mzs.shape[0])
                    # iterate through all the scans in one file
                    for scan in reader:
                        mean = mean + np.array(scan['intensity array'])
                   
                    # perform some signal pre-processing
                    # apply filter before binning
                    #window = 51
                    #order = 2
                    if bin_size is not None:
                        t = 50
                        #savgol_spect = savgol_filter(mean, window, order)
                        corr_spect = white_tophat(mean.astype(np.float64), t)

                        # bin the pre-processed spectrum
                        stats, self.bins, _ = binned_statistic(mzs, corr_spect, 'sum', bins=self.num_bins, range=(low, up))
                    else:
                        stats, self.bins = mean.astype(np.float64), mzs
                    stats[np.isnan(stats)] = 0

                    # add up all bins in the row to get a "total" row value
                    total = sum(stats)
                    row = np.concatenate(([total], stats))    # add total to the beginning of the row
                    file_names.append(file_)
                    rows.append(row)    # write the row to the data for frame
                mzml_count += 1
        
        if bin_size is not None:
            return np.array(rows), np.array([e for e in (self.bins[:-1] + self.bin_size/2)], dtype='float'), np.array([file_names])
        return np.array(rows), np.array(self.bins, dtype='float'), np.array([file_names])

    def grabFileData(self, file_num) -> int:
        i = 0
        for file_ in os.listdir(self.directory):
            print(i, file_)
            if file_.endswith('mzML'):
                if i == file_num:
                    print('check')
                    with mzml.read(rf'{self.directory}\{file_}', use_index=True) as reader:
                        times = np.zeros(len(reader), dtype=np.object)
                        mzs = np.zeros(len(reader), dtype=np.object)
                        intens = np.zeros(len(reader), dtype=np.object)
                        tics = np.zeros(len(reader), dtype=np.object)

                        # iterate through all the scans in one file
                        for j, scan in enumerate(reader):
                            # get m/z values and intensity values from the mzML structure
                            #print(scan['scanList']['scan'][0]['scan start time'])
                            times[j] = scan['scanList']['scan'][0]['scan start time']
                            mzs[j] = scan['m/z array']
                            intens[j] = scan['intensity array']
                            tics[j] = np.sum(intens[j])

                        return {
                            'times' : times,
                            'mzs' : mzs,
                            'intens' : intens,
                            'tics' : tics
                        }
                i += 1
        return 0
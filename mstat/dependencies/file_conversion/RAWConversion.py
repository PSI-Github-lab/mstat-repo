try:
    import sys
    import re
    from subprocess import *
    import os.path
    import time
    import multiprocessing
    from tqdm import tqdm
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from pymsfilereader import MSFileReader
    from scipy.ndimage.morphology import white_tophat
    from scipy.signal import savgol_filter
    from mstat.dependencies.file_conversion.BatchFile import BatchFile
    from mstat.dependencies.file_conversion.MZMLDirectory import MZMLDirectory
except ModuleNotFoundError as exc:
    print(exc)
    print('Install the module via "pip install _____" and then try running the script again')
    input('Press ENTER to leave script...')
    quit()

def get_avg_window(tics, threshold=-1):
    avg = np.mean(tics)
    thres = avg if threshold < 0. else threshold
    #print(avg)
    start = next(x[0] for x in enumerate(tics) if x[1] >= thres)
    reverse = tics[::-1]
    end = -next(x[0] for x in enumerate(reverse) if x[1] >= thres) - 1
    #print(start, end)
    return start, end, thres

def get_summed_spectrum(mzs, intens):
    mean = np.zeros(mzs.shape, dtype='float')
    # iterate through all the scans in one file
    for inten in intens:
        mean = mean + np.array(inten)

    return mzs, mean

def raw_to_numpy_matrix(path : str) -> tuple:
    """
    Create matrix of all scans in a RAW file and returns the matrix along with file metadata. Also, performs some smoothing using a Savitzky-Golay filter.
    Uses pymsfilereader to read RAW files which relies on the Thermo Fisher MSFileReader library.
    See https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.0/faq/data-import-thermo-raw-need-libraries.aspx#downloading-libraries
    Note: this library can be installed on 32- & 64-bit Windows OS (as of 05/27/2022)

    args:
        path    - (str) path to RAW file
    """
    try:
        raw_file = MSFileReader(path)
    except OSError as exc:
        raise exc
    num_scans = raw_file.GetLastSpectrumNumber()

    data = raw_file.GetSummedMassSpectrum([1])[0]    # this function does most of the work
    tic_data = raw_file.GetChroData(startTime=raw_file.StartTime,
                                                endTime=raw_file.EndTime,
                                                massRange1="{}-{}".format(raw_file.LowMass, raw_file.HighMass),
                                                scanFilter="Full ms ")[0]
    times = tic_data[0]
    tics = []
    mzs = np.array(data[0])
    intens = np.zeros((num_scans, len(data[1])))
    for i in range(1, num_scans+1): #, total=num_scans, desc='\nGetting scans from file'):
        data = raw_file.GetMassListFromScanNum(i)[0]
        if mzs.shape[0] != len(data[0]):
            raise ValueError("Scans do not all share the same number of bins.")
        new_intens = np.array(data[1])
        tics.append(sum(new_intens))
        new_intens[np.isnan(new_intens)] = 0
        intens[i-1,:] = new_intens

    metadata = {
        'filename' : str(os.path.basename(path)), 
        'comment1' : str(raw_file.GetComment1()), 
        'comment2' : str(raw_file.GetComment2()), 
        'lowlim' : str(raw_file.GetLowMass()),
        'uplim' : str(raw_file.GetHighMass()), 
        'numscans' : str(num_scans)
    }

    return mzs, intens, np.array(tics), np.array(times), metadata

def raw_to_numpy_array(path : str, sel_region=True, sel_threshold=-1, num_sel_scans=-1, smoothing=True, window=51, order=2) -> tuple:
    """
    Sum all scans of a RAW file and returns cumulative spectrum along with file metadata. Also, performs some smoothing using a Savitzky-Golay filter.
    Uses pymsfilereader to read RAW files which relies on the Thermo Fisher MSFileReader library.
    See https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.0/faq/data-import-thermo-raw-need-libraries.aspx#downloading-libraries
    Note: this library can be installed on 32- & 64-bit Windows OS (as of 05/27/2022)

    args:
        path            - (str) path to RAW file
        sel_scans       - (bool) select region of scans with high TIC for better signal averaging
        sel_threshold   - (float) TIC threshold for detecting high TIC region (if -1, threshold is set to average TIC of all scans)
        num_sel_scans   - (int) number of scans selected in high TIC region (if -1, scans will be selected until falling below threshold)
        smoothing       - (bool) perform smoothing true/false
        window          - (int) window for savgol filter
        order           - (int) polynomial order for savgol filter
    """
    try:
        mzs, intens, tics, _, metadata = raw_to_numpy_matrix(path)
    except Exception as exc:
        raise exc
    
    #get average spectrum in window of interest
    if sel_region:
        start, end, thres = get_avg_window(tics, threshold=sel_threshold)
        if num_sel_scans > 0:
            end = start + num_sel_scans - 1
        if end <= -1:
            mzs, sum_intens = get_summed_spectrum(mzs, intens[start:])
        else:
            mzs, sum_intens = get_summed_spectrum(mzs, intens[start:end+1])
        metadata['scanselect'] = True
    else:
        start, end, thres = 0, -1, 0.0
        mzs, sum_intens = get_summed_spectrum(mzs, intens)
        metadata['scanselect'] = False
    metadata['startscan'] = start
    metadata['endscan'] = end
    metadata['threshold'] = thres
    
    # apply filter
    if smoothing:
        savgol_spect = savgol_filter(sum_intens, window, order)
        metadata['smoothing'] = True
        return mzs, savgol_spect, metadata
    metadata['smoothing'] = False
    return mzs, sum_intens, metadata

def run_single_batch(directory, file):
    """
    Runs external batch script using the ThermoRawFileParser tool to convert a single RAW to MZML format.
    See https://github.com/compomics/ThermoRawFileParser
    Note: this tool can only be installed on 64-bit Windows OS
    """
    batch = BatchFile(r'dependencies\file_conversion\ConvertSingleRAW.bat')
    status, _, errors = batch.run([rf'{directory}\{file}', rf'{directory}\MZML'], False)

    if status != 0:
        print("ERROR: Batch conversion process terminated with the following errors")
        print(f"{errors}")
        quit()

def run_batch_conversion(directory, max_jobs = 10):
    """
    Wrapper around runSingleBatch to run RAW to MZML conversion on many files.
    Files are grouped into multiple batches which run on separate threads (via multiprocessing module).
    """
    print(f"STATUS: Running RAW conversion in directory {directory}")
    jobs = []
    batch = 1
    n_jobs = len(jobs)
    for file in os.listdir(directory):
        if file.endswith('raw'):
            # run the batch script
            process = multiprocessing.Process(target=run_single_batch, args=(directory, file))

            # flush jobs when max conversion jobs number is reached
            if n_jobs >= max_jobs:
                print(f"STATUS: Running conversion batch #{batch}")
                for j in jobs:
                    j.start()

                for j in tqdm.tqdm(jobs, desc="Batch Progress"):
                    j.join()

                jobs = []
                batch += 1

            jobs.append(process)
            n_jobs = len(jobs)
    
    # flush remaining conversion jobs
    print("STATUS: Running conversion remaining batch")
    for j in jobs:
        j.start()

    for j in tqdm.tqdm(jobs, desc="Batch Progress"):
        j.join()

def raw_to_mzml(directories, run_batch, verbose=False):
    """
    Run batch file for converting RAW files to mzML format for multiple directories using runBatchConversion
    """
    # go through given directories and create MZML files for each discovered class
    mzmlDirectories = []
    classes = []

    for directory in directories:
        start_time = time.time()
        # get class name from directory path and make new folder
        class_name = re.findall(r'\\[a-zA-Z]*\\', directory[::-1])[0][::-1][1:-1]
        classes.append(class_name)
        try:
            os.mkdir(rf'{directory}\MZML')
        except FileExistsError:
            if verbose:
                print(f'STATUS: {directory}/MZML directory already exists...')

        if run_batch == 1:
            run_batch_conversion(directory, 10)
        else:
            print("STATUS: Not running RAW conversion in directories")

        # move to working with mzML files to convert data to CSV format
        data = MZMLDirectory(rf'{directory}\MZML')
        lims = [1] #data.checkMZLims()
        if not lims:
            print("ERROR: No data found...")
            quit()
        else:
            print('STATUS: Proper data found.')
            #print(f"STATUS: M/Z limits found in directory are {lims}")
            mzmlDirectories.append(data)

        print(f"--- Ran in {time.time() - start_time} seconds ---") 

    return mzmlDirectories, classes

def mzml_to_csv(mzmlDirectories, target, classes, csv_name, bin_size, low_lim, up_lim, tt_split=0.0, verbose=False):
    """
    convert MZML to CSV via the pandas module
    """
    #start_time = time.time()
    print("STATUS: Writing CSV file...")

    frames = []
    for data, class_ in tqdm(zip(mzmlDirectories, classes), desc="CSV Progress", total=len(mzmlDirectories)):
        df = data.createFrame(bin_size, low_lim, up_lim)
        if df.empty:
            print('ERROR: No MZML files found')
            quit()

        label = [class_] * len(df.values)
        df.insert(0, 'label', label, True)
        frames.append(df)
    #print("--- Ran in %s seconds ---" % (time.time() - start_time))   

    final_data = pd.concat(frames)

    # convert dataframe to csv
    if tt_split != 0.0:
        train, test = train_test_split(final_data, test_size=tt_split, shuffle=True)
        train.to_csv(f'{target}/{csv_name}_Train.csv', index=True)
        test.to_csv(f'{target}/{csv_name}_Test.csv', index=True)
    else:
        final_data.to_csv(f'{target}/{csv_name}.csv', index=True)
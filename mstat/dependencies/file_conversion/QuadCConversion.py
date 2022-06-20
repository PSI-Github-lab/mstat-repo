import numpy as np
import os

def white_noise(shape : tuple, magnitude : float):
    try:
        xs = shape[0]
    except Exception:
        return magnitude * np.random.rand(shape[1])
    try:
        ys = shape[1]
    except Exception:
        return magnitude * np.random.rand(shape[0])
    return magnitude * np.random.rand(xs, ys)

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def quadc_to_numpy_matrix(path : str):
    # extracting data from a single file (tab delimited)
    if '.txt' not in path.lower():
        raise ValueError('Wrong file type (Need to give a .txt file)')

    try:
        with open(path, 'r') as file:
            lines = file.readlines()
    except:
        raise FileNotFoundError(f'Error opening {os.path.basename(path)}')

    file_name = lines[0].strip()

    data_check = False
    num_scans = 0
    num_pairs = 0
    pair_count = 0
    ret_times = []
    mzs = []
    bins = []
    intens = []
    values = []
    for line in lines[1:]:
        entries = line.strip().split('\t')
        if not has_numbers(entries[0]) and not data_check:
            num_scans += 1
            ret_times.append(float(entries[3]))
            num_pairs = int(''.join(filter(str.isdigit, entries[-1])))
            pair_count = 0
            data_check = True
        elif data_check:
            pair_count += 1
            bins.append(float(entries[0]))
            values.append(float(entries[1]))

            if bins and pair_count >= num_pairs:
                mzs.append(np.array(bins))
                intens.append(np.array(values))
                bins = []
                values = []
                data_check = False
        else:
            raise ValueError(f'Unknown error occurred in line\n{line}')

    mzs = np.array(mzs)
    intens = np.array(intens)

    metadata = {
        'filename' : str(os.path.basename(file_name)), 
        'comment1' : '', 
        'comment2' : '', 
        'lowlim' : str(mzs.min()),
        'uplim' : str(mzs.max()), 
        'numscans' : str(num_scans)
    }

    return mzs, intens, np.sum(intens, axis=1), np.array(ret_times), metadata

def quadc_to_numpy_array(path : str, noise_coef=0.0, mca_mode=False):
    try:
        mzs, intens, tics, _, metadata = quadc_to_numpy_matrix(path)
    except Exception as exc:
        raise exc

    if mca_mode:
        spectrum = intens[-1]
    else:
        spectrum = np.sum(intens, axis=0)
    
    if noise_coef != 0.0:
        spectrum = spectrum + white_noise(shape=spectrum.shape, magnitude=noise_coef*np.mean(spectrum))
        

    return mzs[0], spectrum, metadata
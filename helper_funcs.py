import glob
import enum
import pandas as pd
import numpy as np
import os
from scipy.stats import binned_statistic

def checkMZML(path, num_raw):
    num_mzml = len(glob.glob1(f'{path}/MZML', "*.mzml"))
    if num_mzml == num_raw:
        return 1
    elif num_mzml == 0:
        return 0
    return 2

def checkNPY(path, num_raw):
    for file in os.listdir(path):
        if file.endswith('npy'):
            with open(rf'{path}\{file}', 'rb') as f:
                intens = np.load(f)

            num_rows = intens.shape[0]
            if num_rows == num_raw:
                return 1
            elif num_rows == 0:
                return 0
            return 2
    return 2

def calcBins(low, up, bin_size, mz_lims = None):
    if mz_lims is None:
        mz_lims = [-np.inf,np.inf]
    lowlim = max(low, min(mz_lims))
    uplim = min(up, max(mz_lims))
    num_bins = int((uplim - lowlim)/bin_size)
    return np.linspace(lowlim, uplim-bin_size, num_bins), num_bins

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def getFeatureData(frame : pd.DataFrame):
    feature_cols = [col for col in frame if isFloat(col)]     # feature data captured by bins are numeric column labels
    return frame[feature_cols].values

def getLabelData(frame : pd.DataFrame):
    #self.my_encoder = LabelEncoder()
    matrix = frame['label'].values
    return np.array(matrix.tolist())

def get_num_files(path : str, ext : str):
    num_files = 0
    for base, dirs, files in os.walk(path):
        for file in files:
            if ext in file.lower():
                num_files += 1
    return num_files

def sort_tuple_list(tup, entry):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key = lambda x: x[entry])
    return tup

def getTICNormalization(data):
    rows, cols = data.shape
    norm_data = np.empty(0)

    for row in range(rows):
        temp = np.empty(0)
        total = sum(data[row])
        #for element in data[row,1:]:
        temp = data[row].astype('float') / total
        
        try:
            norm_data = np.vstack((norm_data, temp))
        except ValueError:
            norm_data = temp

    return norm_data

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

class ControlStates(enum.Enum):
    READY = 0
    CONVERTING = 1
    PLOTTING = 2
    BUILDING = 3
    LOADING = 4
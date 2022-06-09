from operator import index
from pyteomics import mzml, auxiliary
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import csv

plotting = False

scan = 5
bin_size = 0.5
lowlim = 200
uplim = 1000

filename = 'Salmon_210719123422'

with mzml.read('data/Salmon/2021-07-19/output/'+filename+'.mzml', use_index=True) as reader:
       auxiliary.print_tree(next(reader))
       print(reader[scan]['index'])

       mz_s = reader[scan]['m/z array']
       inten_s = reader[scan]['intensity array']

       if plotting == True:
              fig, ax1 = plt.subplot()
              #ax.stem(mz, inten, markerfmt='')
              ax1.scatter(mz_s, inten_s)

              ax1.set(xlabel='m/z',
                     title='Single Unbinned Spectrum '+str(scan))
              ax1.grid()

              plt.show()

    

    

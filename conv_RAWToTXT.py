from os import name
import sys
from pyteomics import mzml
import numpy as np
from scipy.stats import binned_statistic
import json
from subprocess import *
import os.path
import mmap
import re

from dependencies.file_conversion.BatchFile import BatchFile
from dependencies.file_conversion.MZMLDirectory import MZMLDirectory

def handleStartUpCommands(help_message):
    ''' Take in user commands passed to the script and display help message if requested. '''
    argm = []
    for arg in sys.argv[1:]:
        argm.append(arg)

    if argm[0] == 'help':
        print(help_message)
        quit()

    return argm

help_message = """
Console command: python rawToTXT.py <path/to/RAW/files> <TXT name> <run RAW conversion>
Arguments:
    <path/to/RAW/files>  - (String) Directory with RAW data. Path can be relative or absolute.
    <TXT name>           - (String) Name to add to all TXT created for each RAW file. No need to add .txt at the end.
    <run RAW conversion> - (Boolean) Turn RAW file conversion on or off."""

def main(args):
    '''
    Console command:
    python rawToTXT.py help
    '''
    # handle user commands
    argm = handleStartUpCommands(help_message)

    # define workign directory
    if len(argm) == 0:
        ''' get working directory from user '''
        print("INPUT: Please input working directory which contains relevant RAW files:")
        directory = input()     # should check if input is proper path string
    else:
        directory = argm[0]

    ''' run batch file for converting RAW files to mzML format '''
    mzml_dir = f'{directory}/MZML'
    if int(argm[2]) == 1:
        print(f"STATUS: Running batch file in given directory {directory}")
        try:
            os.mkdir(mzml_dir)
        except FileExistsError:
            print(f'STATUS: {mzml_dir} directory already exists...')
        # run the batch script
        batch = BatchFile(r'dependencies\file_conversion\ConvertRAWinDir.bat')
        status, _, errors = batch.run([directory, mzml_dir, 1, 1])

        # check if the script finished successfully
        if status == 0:
            num_files = len(list(os.listdir(mzml_dir)))
            print(f"STATUS: Batch process completed with no errors and generated {num_files} files in output folder.")
        else:
            print("ERROR: Batch process terminated with the following errors")
            print(errors.decode("utf8"))
            quit()
    else:
        print("STATUS: Not running batch file...")

    ''' move to working with mzML and metadata files to convert data to TXT format '''
    data = MZMLDirectory(mzml_dir)
    lims = data.checkMZLims()
    if len(lims) > 1:
        print("STATUS: Proper metadata found.")
    else:
        print("ERROR: No meta-data found...")
        print(lims)
        quit()


    ''' convert mzML to TXT'''
    if len(argm) == 0:
        # get the name of the TXT file from the user
        print("INPUT: What will the output TXT files be called?")
        txt_name = input()
    else:
        txt_name = argm[1]

    output_dir = f'{directory}/TXT'
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print(f'STATUS: {output_dir} directory already exists...')

    status = data.createTXT(output_dir, txt_name)
    if status == 0:
        print("STATUS: TXT files written. Program completed successfully!")
    else:
        print('ERROR: No MZML files found')
        quit()

if __name__ == '__main__':
    main(0)
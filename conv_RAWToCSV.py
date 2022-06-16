import sys
from subprocess import *
import time
import os.path

from mstat.dependencies.file_conversion.RAWConversion import *
from mstat.dependencies.directory_dialog import *


def handleStartUpCommands(help_message):
    argm = list(sys.argv[1:])
    if argm and argm[0] == 'help':
        print(help_message)
        quit()

    return argm

help_message = """
Console command: python rawToCSV.py <CSV name> <bin size> <min mass> <max mass> <run RAW conversion> <split train/test>
Arguments:
    <CSV name>           - (String) name of the CSV file output. No need to add '.csv' at the end.
    <bin size>           - (Float) bin size for adding all scans together in each RAW file.
    <min mass>           - (Float) max m/z value to consider. Constrained by m/z range observed in RAW files.
    <max mass>           - (Float) min m/z value to consider. Constrained by m/z range observed in RAW files.
    <run RAW conversion> - (Boolean) turn RAW file conversion on or off.
    <split train/test>   - (Float) percentage of data put into testing file"""

def main():
    # handle user commands
    argm = handleStartUpCommands(help_message)
    if len(argm) == 0:
        print("ERROR: Please provide arguments when calling the script. Type 'python rawToCSV.py help' for more information.")
        quit()

    # define working directory
    dirhandler = DirHandler(log_name='rawtocsv', dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()
   
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

    if 'PREV_TARGET' in dirs:
        out_directory = getDirFromDialog("Choose folder for output csv", dirs['PREV_TARGET'])
    else:
        out_directory = getDirFromDialog("Choose folder for output csv")
    if len(out_directory) == 0:
        print('Action cancelled. No directory selected.')
        quit()
    dirhandler.addDir('PREV_TARGET', out_directory)

    dirhandler.writeDirs()

    # define user passed arguments
    csv_name = argm[0]
    bin_size = float(argm[1])
    low_lim = float(argm[2])
    up_lim = float(argm[3])
    tt_split = float(argm[5])

    # run MZML conversion if required, otherwise check that MZML files are present
    mzmlDirectories, classes = raw_to_mzml(in_directories, int(argm[4]))
    
    # combine selected MZML files into a csv file for model training
    mzml_to_csv(mzmlDirectories, out_directory, classes, csv_name, bin_size, low_lim, up_lim, tt_split)

    print("STATUS: CSV file written. Program completed successfully!")
        

if __name__ == '__main__':
    main()
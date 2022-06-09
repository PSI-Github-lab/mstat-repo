import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from helper_funcs import get_num_files
from dependencies.directory_dialog import *
from dependencies.file_conversion.QuadCConversion import *


def main():
    multi_dir_log = 'multidirlog'

    # define working directory
    dirhandler = DirHandler(log_name=multi_dir_log, dir=os.path.dirname(os.path.abspath(__file__)))
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

    dirhandler.writeDirs()

    for dir in in_directories:
        if get_num_files(dir, '.txt') == 0:
            raise ValueError(f'No data files found in {os.path.basename(dir)}')
        print('Viewing:', os.path.basename(dir))

        start_time = time.time()

        mzs = []
        intens = []
        metadata = []
        for file in tqdm(os.listdir(dir), desc='Processed files: ', total=len(os.listdir(dir))):
            if file.endswith('txt'):
                a, b, c = quadc_to_numpy_array(rf'{dir}\{file}', add_noise=False)
                mzs.append(a)
                intens.append(b)
                metadata.append(c)

        mzs = mzs[0][0]     # only need one of these lines for now, but that could change...
        intens = np.array(intens)
        metadata = np.array(metadata)

        file_name = rf'{dir}\{os.path.basename(dir)}.npy'
        with open(file_name, 'wb') as f:
            np.save(f, np.array(intens))
            np.save(f, np.array(mzs))    
            np.save(f, np.array(metadata))

        print(f"--- completed in {time.time() - start_time} seconds ---")

        print('Bins Shape', mzs.shape, 'Counts Shape', intens.shape)
        print(metadata)

    figure = plt.figure()

    i = 0
    plt.scatter(mzs, intens[i])
    plt.title(os.path.basename(metadata[i]['filename']))
    plt.grid()
    plt.xlabel('m/z (Da)')
    plt.ylabel('counts (A.U.)')

    plt.show()


if __name__ == '__main__':
    main()
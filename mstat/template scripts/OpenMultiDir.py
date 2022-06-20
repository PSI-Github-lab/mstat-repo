from dependencies.directory_dialog import *

def main():
    multi_dir_log = 'multidirlog'

    # define working directory
    dirhandler = DirHandler(log_name=multi_dir_log, log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
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


if __name__ == '__main__':
    main()
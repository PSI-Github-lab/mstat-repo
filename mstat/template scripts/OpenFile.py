from dependencies.directory_dialog import *

def main():
    file_log = 'filelog'

    # define working directory
    dirhandler = DirHandler(log_name=file_log, log_folder="mstat/directory logs", dir=os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()
   
    # get file from dialog
    if 'PREV_SOURCE' in dirs:
        in_dir = getFileDialog("Choose a file", "CSV files (*.csv)|*.csv", dirs['PREV_SOURCE'])
    else:
        in_dir = getFileDialog("Choose a file", "CSV files (*.csv)|*.csv")
    if len(in_dir) == 0:
        print('Action cancelled. No directories were selected.')
        quit()
    dirhandler.addDir('PREV_SOURCE', in_dir)

    dirhandler.writeDirs()

if __name__ == '__main__':
    main()
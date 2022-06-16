import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from mstat.dependencies.directory_dialog import *
from subprocess import Popen, PIPE
import tqdm
# from https://thepythoncorner.com/posts/2019-01-13-how-to-create-a-watchdog-in-python-to-look-for-filesystem-changes/
# see https://python-watchdog.readthedocs.io/en/stable/quickstart.html#a-simple-example

new_raw_set = set()
del_raw_set = set()

def on_created(event):
    print(f"{os.path.split(event.src_path)[-1]} has been created")
    new_raw_set.add(event.src_path)
 
def on_deleted(event):
    print(f"{os.path.split(event.src_path)[-1]} has been deleted")
    del_raw_set.add(event.src_path)
 
def on_modified(event):
    print(f"{os.path.split(event.src_path)[-1]} has been modified")
    new_raw_set.add(event.src_path)
 
def on_moved(event):
    print(f"{os.path.split(event.src_path)[-1]} has been moved to {os.path.split(event.dest_path)[-1]}")
    del_raw_set.add(event.src_path)
    new_raw_set.add(event.dest_path)

if __name__ == "__main__":
    wait_time = 10
    debug = False

    # define watchdog pattern event handler to look for RAW files
    patterns = ["*.raw"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    my_event_handler.on_modified = on_modified
    my_event_handler.on_moved = on_moved

    # define working directory
    watchfolder = DirHandler("watchfolder", os.path.dirname(os.path.abspath(__file__)))
    watchfolder.readDirs()
    dirs = watchfolder.getDirs()

    folder_sel_instruction = "Choose folder to monitor for new RAW files"
    if 'PREV_TARGET' in dirs:
        out_directory = getDirFromDialog(folder_sel_instruction, dirs['PREV_TARGET'])
    else:
        out_directory = getDirFromDialog(folder_sel_instruction)
    if len(out_directory) == 0:
        print('Action cancelled. Watchdog is terminated.')
        quit()
    watchfolder.addDir('PREV_TARGET', out_directory)

    watchfolder.writeDirs()

    # find the msconvert file on this computer
    msconvertpath = DirHandler("msconvertpath", os.path.dirname(os.path.abspath(__file__)))
    msconvertpath.readDirs()
    dirs = msconvertpath.getDirs()

    folder_sel_instruction = "Find the folder containing msconvert.exe to process RAW files"
    if 'PREV_TARGET' in dirs:
        if os.path.isfile(dirs['PREV_TARGET'] + r'\\msconvert.exe'):
            msc_directory = dirs['PREV_TARGET']
        else:
            msc_directory = getDirFromDialog(folder_sel_instruction, dirs['PREV_TARGET'])
    else:
        msc_directory = getDirFromDialog(folder_sel_instruction)
    if len(msc_directory) == 0:
        print('Action cancelled. Watchdog is terminated.')
        quit()
    msconvertpath.addDir('PREV_TARGET', msc_directory)

    msconvertpath.writeDirs()

    # create the watchdog observer to keep track of all files in specified directory
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, out_directory, recursive=go_recursively)
    my_observer.start()

    # Start file observation loop 
    try:
        while True:
            print("Watchdog waiting...")
            # wait for some time to allow some new/modified files to accumulate
            time.sleep(wait_time)
            print(f"{len(new_raw_set)} new RAW files since last refresh")
            print(f"{len(del_raw_set)} deleted RAW files since last refresh")

            # convert new files to MZML
            for file_path in tqdm.tqdm(list(new_raw_set.difference(del_raw_set)), desc="Conversion Progress"):
                # create MZML folder in it's directory (if it doesn't exist yet)
                directory = os.path.dirname(file_path)
                try:
                    os.mkdir(rf'{directory}\MZML')
                except FileExistsError:
                    if debug:
                        print(f'STATUS: {directory}/MZML directory already exists...')
                
                # run conversion
                try:
                    cmd = rf'"{msc_directory}\\msconvert.exe" "{file_path}" -o "{directory}\MZML"'
                    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
                    output, errors = p.communicate()
                    p.wait()    # wait for process to terminate
                except Exception as ex:
                    print(f"Conversion of {file_path} failed with exception:\n {ex}")

                if debug:
                    print(output)

                if p.returncode != 0:
                    print("ERROR: msconvert terminated with the following errors")
                    print(f"{errors}")
                
                new_raw_set.remove(file_path)   # remove file name from set once it has been converted
            # delete old MZML files
            # to be completed...
            del_raw_set.clear()
            
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()

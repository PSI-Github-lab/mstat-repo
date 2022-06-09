import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from directory_dialog import *
# from https://thepythoncorner.com/posts/2019-01-13-how-to-create-a-watchdog-in-python-to-look-for-filesystem-changes/
# see https://python-watchdog.readthedocs.io/en/stable/quickstart.html#a-simple-example

def on_created(event):
    print(f"hey, {event.src_path} has been created!")
 
def on_deleted(event):
    print(f"what the f**k! Someone deleted {event.src_path}!")
 
def on_modified(event):
    print(f"hey buddy, {event.src_path} has been modified")
 
def on_moved(event):
    print(f"ok ok ok, someone moved {event.src_path} to {event.dest_path}")


if __name__ == "__main__":
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
    dirhandler = DirHandler(os.path.dirname(os.path.abspath(__file__)))
    dirhandler.readDirs()
    dirs = dirhandler.getDirs()

    folder_sel_instruction = "Choose folder to monitor for new RAW files"
    if 'PREV_TARGET' in dirs:
        out_directory = getDirFromDialog(folder_sel_instruction, dirs['PREV_TARGET'])
    else:
        out_directory = getDirFromDialog(folder_sel_instruction)
    if len(out_directory) == 0:
        print('Action cancelled. Watchdog is terminated.')
        quit()
    dirhandler.addDir('PREV_TARGET', out_directory)

    dirhandler.writeDirs()

    #path = r"C:\Users\Jackson\PSI Files Dropbox\MS Detectors\LTQ\Watchdog_Test"
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, out_directory, recursive=go_recursively)

    my_observer.start()
    try:
        while True:
            print("Watchdog waiting...")
            time.sleep(5)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()

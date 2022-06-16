import os.path
import os
import wx
import wx.lib.agw.multidirdialog as MDD

def getMultDirFromDialog(title_in, default_path=os.path.dirname(__file__)):
    # create file dialog using wxApp
    app = wx.App(0)
    try:
        dlg = MDD.MultiDirDialog(None, title=title_in, defaultPath=default_path,
                                agwStyle=MDD.DD_MULTIPLE|MDD.DD_DIR_MUST_EXIST)
    except IndexError:
        dlg = MDD.MultiDirDialog(None, title=title_in, defaultPath=os.path.dirname(__file__),
                                agwStyle=MDD.DD_MULTIPLE|MDD.DD_DIR_MUST_EXIST)

    if dlg.ShowModal() != wx.ID_OK:
        print("You Cancelled The Dialog!")
        dlg.Destroy()
        return []

    else:
        paths = dlg.GetPaths()
        directories = [
            #path[1].replace('Local Disk (C:)', 'C:').replace('Windows (C:)', 'C:')
            path[1][path[1].find('C:'):].replace('C:)', 'C:')
            for path in enumerate(paths)
        ]

    dlg.Destroy()
    app.MainLoop()

    return directories

def getDirFromDialog(message_in, default_path=os.path.dirname(__file__)):
    # create file dialog using wxApp
    app = wx.App(0)
    try:
        dlg = wx.DirDialog(None, message=message_in, defaultPath=default_path)
    except IndexError:
        dlg = wx.DirDialog(None, message=message_in, defaultPath=os.path.dirname(__file__))

    if dlg.ShowModal() != wx.ID_OK:
        print("You Cancelled The Dialog!")
        dlg.Destroy()
        return []

    else:
        path = dlg.GetPath()
        directory = path[path.find('C:'):].replace('C:)', 'C:')

    dlg.Destroy()
    app.MainLoop()

    return directory

def getFileDialog(message_in, pattern, default_path=os.path.dirname(__file__)):
    app = wx.App(0)

    try:
        fileDialog =  wx.FileDialog(None, message_in, defaultDir=default_path, wildcard=pattern,
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    except IndexError:
        fileDialog =  wx.FileDialog(None, message_in, defaultDir=os.path.dirname(__file__), wildcard=pattern,
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

    if fileDialog.ShowModal() == wx.ID_CANCEL:
        fileDialog.Destroy()
        return []     # the user changed their mind
    else:
        # Proceed loading the file chosen by the user
        pathname = fileDialog.GetPath()
        pathname = pathname[pathname.find('C:'):].replace('C:)', 'C:')

    fileDialog.Destroy()
    app.MainLoop()
    
    return pathname

class DirHandler:
    def __init__(self, log_name='dirlog', dir=os.path.dirname(__file__)):
        self.cur_directory = dir
        self.dirs = {}
        self.log_name = log_name

        try:
            with open(self.cur_directory + r'\\' + f"{self.log_name}.log", 'r+') as dirlog:
                pass
        except Exception:
            with open(self.cur_directory + r'\\' + f"{self.log_name}.log", 'w+') as dirlog:
                pass
    
    def readDirs(self):
        with open(self.cur_directory + r'\\' + f"{self.log_name}.log", 'r+') as dirlog:
            lines = dirlog.readlines()
            if len(lines) > 0 and lines[0].split(' ')[0] in [
                'PREV_SOURCE',
                'PREV_TARGET',
            ]:
                for line in lines:
                    line = line.strip()
                    elements = line.split(' ')
                    self.dirs[elements[0]] = ' '.join(elements[1:])

    def addDir(self, name, dir):
        self.dirs[name] = dir

    def getDirs(self):
        return self.dirs

    def writeDirs(self):
        with open(self.cur_directory + r'\\' + f"{self.log_name}.log", 'w+') as dirlog:
            for item in self.dirs.items():
                dirlog.write(f"{item[0]} {item[1]}\n")
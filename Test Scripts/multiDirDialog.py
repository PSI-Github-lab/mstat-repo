####### Retrieve a list of directories with wxPython-Phoenix   - tested on python3.5
### installation instruction for wxPython-Phoenix  : https://wiki.wxpython.org/How%20to%20install%20wxPython#Installing_wxPython-Phoenix_using_pip
### modified from : https://wxpython.org/Phoenix/docs/html/wx.lib.agw.multidirdialog.html
import os
import wx
import wx.lib.agw.multidirdialog as MDD

# Our normal wxApp-derived class, as usual
app = wx.App(0)
dlg = MDD.MultiDirDialog(None, title="Choose folders containing raw files for conversion", defaultPath=os.path.dirname(__file__),  # defaultPath="C:/Users/users/Desktop/",
                         agwStyle=MDD.DD_MULTIPLE|MDD.DD_DIR_MUST_EXIST)

if dlg.ShowModal() != wx.ID_OK:
    print("You Cancelled The Dialog!")
    dlg.Destroy()

else:
    paths = dlg.GetPaths()
    print(paths)

    #Print directories' path and files 
    for path in enumerate(paths):
        print(path[1])
        directory= path[1].replace('Local Disk (C:)','C:')
        print(directory)
        for file in os.listdir(directory):
            print(file)

dlg.Destroy()
app.MainLoop()
print('hello')
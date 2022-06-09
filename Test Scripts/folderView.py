'''import os
import tkinter as tk
from tkinter import ttk

root=tk.Tk()
root.geometry('600x300')

f=tk.Frame(root)
tv=ttk.Treeview(f,show='tree')
ybar=tk.Scrollbar(f,orient=tk.VERTICAL,
                  command=tv.yview)
tv.configure(yscroll=ybar.set)

directory='ltq_data'
tv.heading('#0',text='Dir：'+directory, anchor='w')
path=os.path.abspath(directory)
node=tv.insert('','end',text=path,open=True)
def traverse_dir(parent,path):
    for d in os.listdir(path):
        full_path=os.path.join(path,d)
        isdir = os.path.isdir(full_path)
        id=tv.insert(parent,'end',text=d,open=False)
        if isdir:
            traverse_dir(id,full_path)
traverse_dir(node,path)
ybar.pack(side=tk.RIGHT,fill=tk.Y)
tv.pack()
f.pack()
root.mainloop()'''

#!/usr/bin/env python3

"""
ZetCode Tkinter tutorial

In this script, we use the pack manager
to position two buttons in the
bottom-right corner of the window.

Author: Jan Bodnar
Website: www.zetcode.com
"""

import os
import tkinter as tk
from tkinter import Tk, RIGHT, LEFT, BOTH, RAISED, VERTICAL
from tkinter.ttk import Frame, Button, Style, Treeview, Scrollbar

class Example(Frame):
    def __init__(self):
        super().__init__()

        self.initUI()
    
    def traverse_dir(self, tv, parent, path):
        for d in os.listdir(path):
            full_path=os.path.join(path,d)
            isdir = os.path.isdir(full_path)
            id = tv.insert(parent,'end',text=d,open=False)
            if isdir:
                self.traverse_dir(tv, id, full_path)

    def initUI(self):
        self.master.title("Folder View")
        self.style = Style()
        self.style.theme_use("default")

        frame = Frame(self, relief=RAISED, borderwidth=1)
        frame.pack(fill=BOTH, expand=True)

        self.pack(fill=BOTH, expand=True)

        tv1 = Treeview(frame, show='tree')
        ybar1 = Scrollbar(frame, orient = VERTICAL, command=tv1.yview)
        tv1.configure(yscroll=ybar1.set)

        directory1='ltq_data'
        #tv1.heading('#0',text='Dir：'+ directory1, anchor='w')
        path = os.path.abspath(directory1)
        node = tv1.insert('','end',text=path,open=True)
        self.traverse_dir(tv1, node, path)

        tv1.pack(side=LEFT, padx=40, pady=5)
        ybar1.pack(side=LEFT, fill=tk.Y)

        tv2 = Treeview(frame, show='tree')
        ybar2 = Scrollbar(frame, orient=VERTICAL, command=tv2.yview)
        tv2.configure(yscroll=ybar2.set)

        directory2='csv_output'
        #tv2.heading('#0',text = 'Dir：' + directory2, anchor='w')
        path = os.path.abspath(directory2)
        node = tv2.insert('', 'end', text=path,open=True)
        self.traverse_dir(tv2, node, path)

        tv2.pack(side=RIGHT, padx=40, pady=5)
        ybar2.pack(side=RIGHT, fill=tk.Y)

        closeButton = Button(self, text="Close")
        closeButton.pack(side=RIGHT, padx=5, pady=5)
        okButton = Button(self, text="OK")
        okButton.pack(side=RIGHT, padx=5, pady=5)


def main():

    root = Tk()
    root.geometry("600x300")
    app = Example()
    root.mainloop()


if __name__ == '__main__':
    main()
import os
import tkinter.filedialog
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

WIKIPEDIA_HOME_PATH = HOME_PATH + "wikipedia/"
WIKIPEDIA_DATA_PATH = WIKIPEDIA_HOME_PATH + "texts/"

UNALTERED_FOLDER_NAME = "u"

def select_file(init_dir=HOME_PATH, choose_file=True):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    if choose_file:
        filename = askopenfilename(initialdir=init_dir,
                                   defaultextension="txt")  # show an "Open" dialog box and return the path to the selected file
        return filename
    else:
        foldername = tkinter.filedialog.askdirectory(initialdir=init_dir)
        return foldername

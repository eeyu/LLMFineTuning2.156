import os
import tkinter.filedialog
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import torch

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

WIKIPEDIA_HOME_PATH = HOME_PATH + "Datasets/" + "wikipedia/"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

nomi_write_token = "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"
nomi_read_token = "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"
annie_read_token = "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"

glm_checkpoint = "THUDM/chatglm3-6b"
llama_checkpoint = "meta-llama/Llama-2-13b-chat-hf"

wikipedia_dataset_checkpoint = "yu-nomi/wikipedia-2.156"
standards_dataset_checkpoint = "yu-nomi/standards-2.156"

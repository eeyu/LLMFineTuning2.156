from datasets import load_dataset
import json
import os


import paths

# https://huggingface.co/docs/datasets/nlp_load
# https://huggingface.co/docs/datasets/package_reference/loading_methods

dataset_checkpoint = paths.standards_dataset_checkpoint

# folder = "train_data/"
folder = paths.select_file(init_dir=paths.WIKIPEDIA_DATA_PATH, choose_file=False)
all_files = []
for path, subdirs, files in os.walk(folder):
    for name in files:
        all_files.append(os.path.join(path, name))
# files = [folder + "/" + x for x in files if ".txt" in x]

dataset = load_dataset("text", data_files={'train': all_files}, sample_by="line")
dataset.push_to_hub(dataset_checkpoint, token=paths.nomi_write_token)
import os
import pandas as pd
import torch
from torch.utils.data import random_split

from data_loader import load_acc_data_with_label, get_corresponding_gyr_path

if __name__ == "__main__":
    samples = []
    labels = []
    acc_data_list = load_acc_data_with_label()
    for acc_path, voice_number in acc_data_list.items():
        gyr_path = get_corresponding_gyr_path(acc_path)
        samples.append([acc_path, gyr_path])
        labels.append(voice_number)

    # Construct file path by concatenating fold and file name
    data_dict = {'relative_path': samples, 'classID': labels}
    df = pd.DataFrame.from_dict(data_dict)
    print(df.head(5))

    data_path = ""

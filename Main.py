import os
import pandas as pd
import torch
from torch.utils.data import random_split

from IMUClassifier import IMUClassifier
from IMUDataset import IMUDS
from ResNet import resnet18
from TrainUtil import training, inference
from data_loader import load_acc_data_with_label, get_corresponding_gyr_path
from read_data import noise_computation

if __name__ == "__main__":
    paths = ["files_individual/files_2_4_6_8", "files_individual/files_0_1"]

    samples = []
    labels = []
    acc_data_dict = load_acc_data_with_label(paths)

    for acc_path, voice_number in acc_data_dict.items():
        gyr_path = get_corresponding_gyr_path(acc_path)
        samples.append([acc_path, gyr_path])
        labels.append(voice_number)

    # Construct file path by concatenating fold and file name
    data_dict = {'relative_path': samples, 'classID': labels}
    df = pd.DataFrame.from_dict(data_dict)

    print(df.head(5))

    noise_path = "files_individual/noise/"
    noise_acc, noise_gyr = noise_computation(noise_path + "acc_1_999_999.txt", noise_path + "gyr_1_999_999.txt")
    imu_ds = IMUDS(df, noise_acc, noise_gyr)

    # Random split of 80:20 between training and validation
    num_items = len(imu_ds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(imu_ds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

    # set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Use CNN
    # myModel = IMUClassifier()
    # Use ResNet
    myModel = resnet18()
    myModel = myModel.to(device)

    # Training Model
    num_epochs = 15
    training(myModel, train_dl, val_dl, num_epochs)

    # Inference
    inference(myModel, val_dl)
    torch.save(myModel,"model/cnn_net.pth")
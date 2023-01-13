import os
import pandas as pd
import torch
from torch.utils.data import random_split

from IMUClassifier import IMUClassifier
from IMUDataset import IMUDS
from SENet import SENet
from MobileNetV2 import MobileNetV2
from DenseNet import DenseNet
from ResNet import resnet18
from TrainUtil import training, inference
from data_loader import load_acc_data_with_label, get_corresponding_gyr_path
from read_data import noise_computation

if __name__ == "__main__":
    paths = ["files_train/signal_data_new/files_0", "files_train/signal_data_new/files_1", "files_train/signal_data_new/files_2", 
    "files_train/signal_data_new/files_3", "files_train/signal_data_new/files_4", "files_train/signal_data_new/files_5", 
    "files_train/signal_data_new/files_6", "files_train/signal_data_new/files_7", "files_train/signal_data_new/files_8", 
    "files_train/signal_data_new/files_9"]

    samples = []
    labels = []
    data_dict = load_acc_data_with_label(paths)

    for data_path, voice_number in data_dict.items():
        samples.append(data_path)
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
    torch.manual_seed(1234)
    train_ds, val_ds = random_split(imu_ds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    # set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # myModel = IMUClassifier()
    # Use SENet
    # myModel = DenseNet(input_channel=1, n_classes=10, 
    #         growthRate=12, depth=20, reduction=0.5, bottleneck=True)
    myModel = torch.load("model/dense_net.pth")
    myModel = myModel.to(device)

    # Training Model
    num_epochs = 30
    print("start training")
    training(myModel, train_dl, val_dl, num_epochs)

    # Inference
    correlation_matrix = inference(myModel, val_dl, is_correlation=True)
    print(correlation_matrix)

    torch.save(myModel,"model/dense_net.pth")

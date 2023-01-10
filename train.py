# Import required modules
import os
import math
import json
import random
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch.nn import init
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

##################################################
# Global settings and data setup                 #
##################################################

# Select torch device (GPU preferred)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# Create directories (adjust if needed)
TRAIN_DIR = '../input/dl-challenge-data/data/data/train/'
TEST_DIR = '../input/dl-challenge-data/data/data/test/'

TRAIN_JSON = '../input/dl-challenge-data/train.json'
TEST_JSON = '../input/dl-challenge-data/test.json'

# Load train and test JSON files
f_train = open(TRAIN_JSON)
train_json = json.load(f_train)
f_train.close()

f_test = open(TEST_JSON)
test_json = json.load(f_test)
f_test.close()

# Convert train_json to a dataframe with filename and label
train = pd.DataFrame(
    [{"filename": filename, "label": label} for (filename, label) in train_json.items()])

# Convert test_json to a dataframe with filename and empty label
test = pd.DataFrame(
    [{"filename": filename, "label": label} for (filename, label) in test_json.items()])
test['label'] = 0

# Add file path to dataframe for our train and test data
train['file_path'] = TRAIN_DIR + train['filename']
train = train[['filename', 'file_path', 'label']]

test['file_path'] = TEST_DIR + test['filename']
test = test[['filename', 'file_path', 'label']]

# Convert training labels to int temporarily for training
# (later back to string for submission)
train['label'] = pd.to_numeric(train['label'])
print(type(train['label'][0]))

# We have to adjust the labels -1 for the model to train because
# cross entropy expects labels from [0, num_labels - 1], later we adjust this
train['label'] = train['label'] - 1

print("Completed setup")


##################################################
# Define classes and functions                   #
##################################################

# List audio files in directory
def list_files(train_dir, test_dir):
    train_clips = os.listdir(train_dir)
    print("No. of .wav files in train folder = ", len(train_clips))

    test_clips = os.listdir(test_dir)
    print("No. of .wav files in test folder = ", len(test_clips))


# Define plotting function - for visualisation of a spectrogram or MFCC
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


# Define processing functions in class AudioUtil
class AudioUtil():
    # Load audio file, return the signal as a tensor and the sample rate
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # Convert to desired number of channels
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    # Resample one channel at a time
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    # Pad (or truncate) the signal to fixed length 'max_ms' in milliseconds
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # Shifts signal to the left or right by some percent
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # Generate Spectrogram
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80  # reasonable number is 80, source: pytorch.org

        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(
            sig)  # spec has shape [channel, n_mels, time]

        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)  # convert to dB
        return (spec)

    # Generate MFCC (configured with parameters for final model)
    @staticmethod
    def mfcc_transform(aud, sample_rate=16000, n_mfcc=256):
        sig, sr = aud
        top_db = 80
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=128,
            melkwargs={
                "n_fft": 1024,
                "n_mels": 128,
                "hop_length": 512,
                "mel_scale": "htk",
            },
        )(sig)

        # Convert to decibels
        mfcc_transform = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(mfcc_transform)

        return (mfcc_transform)

    # Augment spectrogram with frequency- and time masks
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


##################################################
# Setup data loader and batches                  #
##################################################

# Define data loader (creating MFCCs)
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.duration = 4000  # Optimal: 4000.
        self.sr = 16000
        self.channel = 2
        self.shift_pct = 0.3  # Optimal: 0.3.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, 'file_path']  # We have the path defined in the dataframe
        class_id = self.df.loc[idx, 'label']  # Get Label

        # Apply each processing step in order
        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        mfcc = AudioUtil.mfcc_transform(shift_aud, sample_rate=16000, n_mfcc=128)

        return mfcc, class_id


# Create train and test data loaders - train on all data and predict the unlabelled data
train_ds = SoundDS(train, TRAIN_DIR)
test_ds = SoundDS(test, TEST_DIR)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

print("Created data loader")


# Create training (80%) and validation (20%) sets (skipping for final prediction)
# --------------------------------------------------------------------------------
# myds = SoundDS(train, TRAIN_DIR) # Call data loader with our train dataframe and path (path not used atm)
# num_items = len(myds)
# num_train = round(num_items * 0.8)
# num_val = num_items - num_train
# train_ds, val_ds = torch.utils.data.random_split(myds, [num_train, num_val])
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
# val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
# --------------------------------------------------------------------------------

##################################################
# Setup the audio classification model           #
##################################################

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Sixth Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=183)  # Set output labels

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # Forward pass computations
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device
print("Model setup completed")
print("Starting training")


##################################################
# Training the model                             #
##################################################

def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()  # For multi class
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimal: Adam and 0.001.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get input features and target labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print('-------')
        print(acc)
        print(avg_loss)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        print('-------')

    print('Finished Training')


num_epochs = 14  # Optimal: 14 epochs.
training(myModel, train_dl, num_epochs)


##################################################
# Save / load the model state                    #
##################################################

# If we need to save the model
# torch.save(myModel.state_dict(), "model-ver1.pth") # Edit path to your dir

# If we need to load the model again, run following code
# -------------------------------------------------------
# myModel = AudioClassifier()
# myModel.load_state_dict(torch.load('../input/dl-challenge-data/model-ver1.pth'))
# myModel.eval()
# myModel = myModel.to(device)
# next(myModel.parameters()).device
# -------------------------------------------------------

##################################################
# Generate predictions with trained model        #
##################################################

def predictions(model, test_dl):
    all_predictions = []

    with torch.no_grad():
        for data in test_dl:
            inputs = data[0].to(device)

            # Normalize
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            all_predictions.append(prediction)

        return all_predictions


print("Generating predictions")
all_predictions = predictions(myModel, test_dl)

# Gather predictions
final_predictions = []
for batch in all_predictions:
    for i in batch:
        final_predictions.append(i.item())

# Update labels in our test dataframe
test['label'] = final_predictions

# Update labels +1 (because we reduced them earlier)
test['label'] = test['label'] + 1

# Convert labels to str format for submission
test['label'] = test['label'].apply(str)

# Create dictionary with names and labels
predicted_dict = {}
for i, row in test.iterrows():
    predicted_dict[row['filename']] = row['label']

# Generate predictions json file
with open("predicted.json", "w") as outfile:
    json.dump(predicted_dict, outfile)
print("File output, I am all done!")
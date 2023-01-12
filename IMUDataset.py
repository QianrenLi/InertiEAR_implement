from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

# ----------------------------
# IMU Dataset
# ----------------------------
from data_loader import generate_signal, convert_to_spec, pad_trunc, get_silence_noise


class IMUDS(Dataset):
    def __init__(self, df, acc_noise, gyr_noise):
        self.df = df
        self.max_len = 800
        self.acc_noise = acc_noise
        self.gyr_noise = gyr_noise

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Get ith item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        imu_data_files = self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        acc_path = imu_data_files[0]
        gyr_path = imu_data_files[1]
        signal = generate_signal(acc_path, gyr_path, self.acc_noise, self.gyr_noise)
        signal = pad_trunc(signal, self.max_len)
        sgram = convert_to_spec(signal)

        return sgram, class_id

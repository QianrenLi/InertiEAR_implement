from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio


# ----------------------------
# IMU Dataset
# ----------------------------
from data_loader import generate_signal, convert_to_spec


class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)

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
        signal = generate_signal(acc_path, gyr_path)
        sgram = convert_to_spec(signal)

        aud = AudioUtil.open(audio_file)
        # re_noise_aud = AudioUtil.denoise(aud)
        re_channel = AudioUtil.rechannel(aud, self.channel)

        dur_aud = AudioUtil.pad_trunc(re_channel, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return sgram, class_id
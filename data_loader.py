import os

import matplotlib.pyplot as plt
import numpy
import torch

import read_data
from read_data import signal_read, remove_mean_value, segmentation_handle, pre_processing


def load_acc_data_with_label(paths):
    data_dict = {}
    for path in paths:
        for cur_file_name in os.listdir(path):
            if cur_file_name.startswith("acc"):
                voice_number = cur_file_name.replace(".txt", "").replace("acc_1_", "").replace("gyr_1_", "").split("_")[0]
                voice_number = int(voice_number)
                if voice_number <= 9:
                    data_dict[path + "/" + cur_file_name] = voice_number

    return data_dict


def get_corresponding_gyr_path(acc_data_path):
    gyr_data_path = acc_data_path.replace("acc", "gyr")
    return gyr_data_path


def get_silence_noise(acc_noise="./files_individual/noise/acc_1_999_999.txt",
                      gyr_noise="./files_individual/noise/gyr_1_999_999.txt"):
    acc_noise = acc_noise.replace("acc", "silence")
    gyr_noise = gyr_noise.replace("gyr", "silence")
    return acc_noise, gyr_noise


def pad_trunc(signal_aud, max_len):
    if len(signal_aud) < max_len:
        signal_aud = numpy.pad(signal_aud, (0, max_len - len(signal_aud)), 'constant')
    else:
        signal_aud = signal_aud[:max_len]
    return numpy.array(signal_aud)


def generate_signal(acc_data_path, gyr_data_path):
    acc_t, acc_xyz = signal_read(acc_data_path)
    gyr_t, gyr_xyz = signal_read(gyr_data_path)

    acc_xyz = remove_mean_value(acc_xyz)
    gyr_xyz = remove_mean_value(gyr_xyz)

    h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, 400)

    acc_t_idx = numpy.array([[0, -1]])
    gyr_t_idx = numpy.array([[0, -1]])
    signal = pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t)
    return signal[0]


def convert_to_spec(signal):
    signal = torch.Tensor([signal.tolist()])
    sgram = read_data.AudioUtil.spectro_gram((signal, 800), n_fft=256,win_length=8)
    sgram_numpy = sgram.numpy()
    sgram_tensor = torch.Tensor(sgram_numpy)
    # print(sgram_numpy.shape)
    # plt.imshow(sgram_numpy.transpose(1, 2, 0))
    # plt.show()
    return sgram_tensor


if __name__ == '__main__':
    for file_name in os.listdir("files_individual"):
        print(file_name)
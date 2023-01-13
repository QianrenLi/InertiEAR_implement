import os

import matplotlib.pyplot as plt
import numpy
import torch

import read_data
from read_data import signal_read, remove_mean_value, segmentation_handle, pre_processing,noise_computation


def load_acc_data_with_label(paths):
    data_dict = {}
    noise_acc, noise_gyr = noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")
    for path in paths:
        label = int(path.split("_")[-1])
        for cur_file_name in os.listdir(path):
            data_dict[path + "/" + cur_file_name] = label
    return data_dict


def get_corresponding_gyr_path(acc_data_path):
    gyr_data_path = acc_data_path.replace("acc", "gyr")
    return gyr_data_path


def get_silence_noise(acc_noise_path="./files_individual/noise/acc_1_999_999.txt",
                      gyr_noise_path="./files_individual/noise/gyr_1_999_999.txt"):
    noise_acc, noise_gyr = read_data.noise_computation(acc_noise_path, gyr_noise_path)
    return noise_acc, noise_gyr


def pad_trunc(signal_aud, max_len):
    if len(signal_aud) < max_len:
        signal_aud = numpy.pad(signal_aud, (0, max_len - len(signal_aud)), 'constant')
    else:
        signal_aud = signal_aud[:max_len]
    return numpy.array(signal_aud)


def generate_signal(acc_data_path, gyr_data_path, acc_noise, gyr_noise):
    acc_t, acc_xyz = signal_read(acc_data_path)
    gyr_t, gyr_xyz = signal_read(gyr_data_path)

    acc_xyz = remove_mean_value(acc_xyz)
    gyr_xyz = remove_mean_value(gyr_xyz)


    h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, 400)
    segmentation_idx = h_seg.segmentation(2000, acc_noise, gyr_noise)

    signal = pre_processing(acc_xyz, gyr_xyz, segmentation_idx, segmentation_idx, acc_t, gyr_t, acc_noise, gyr_noise)

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
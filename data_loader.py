import os

import matplotlib.pyplot as plt
import numpy

from read_data import signal_read, remove_mean_value, segmentation_handle, pre_processing


def load_acc_data_with_label(path='./files_individual_0_4/files'):
    data_list = {}
    for file_name in os.listdir(path):
        if file_name.startswith("acc"):
            voice_number = file_name.replace(".txt", "").replace("acc_1_", "").replace("gyr_1_", "").split("_")[0]
            data_list[path + "/" + file_name] = voice_number

    return data_list


def get_corresponding_gyr_path(acc_path):
    gyr_path = acc_path.replace("acc", "gyr")
    return gyr_path


def get_silence_noise(acc_noise="./files_individual_0_4/noise/acc_1_999_999.txt",
                      gyr_noise="./files_individual_0_4/noise/gyr_1_999_999.txt"):
    acc_noise = acc_noise.replace("acc", "silence")
    gyr_noise = gyr_noise.replace("gyr", "silence")
    return acc_noise, gyr_noise


def generate_signal(acc_path, gyr_path):
    acc_t, acc_xyz = signal_read(acc_path)
    gyr_t, gyr_xyz = signal_read(gyr_path)

    acc_xyz = remove_mean_value(acc_xyz)
    gyr_xyz = remove_mean_value(gyr_xyz)

    h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, 400)

    acc_t_idx = numpy.array([[0, -1]])
    gyr_t_idx = numpy.array([[0, -1]])
    signal = pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t)
    plt.plot(signal[0])
    plt.show()


if __name__ == '__main__':
    acc_data_list = load_acc_data_with_label()
    i = 0
    for acc_path, label in acc_data_list.items():
        gyr_path = get_corresponding_gyr_path(acc_path)
        generate_signal(acc_path, gyr_path)
        i = i + 1
        if i > 10:
            break

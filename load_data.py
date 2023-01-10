# read all the files in files_0_4/file
import os


def load_data_form_path(path='./files_0_4/files/'):
    data_list = {}
    for file_name in os.listdir(path):
        voice_number = file_name.replace(".txt", "").replace("acc_1_", "").replace("gyr_1_", "")
        if voice_number in data_list:
            data_list[voice_number].append(path + file_name)
        else:
            data_list[voice_number] = [path + file_name]

    return data_list

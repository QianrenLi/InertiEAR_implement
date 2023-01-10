import os


def load_acc_data_with_label(path='./files_0_4/files'):
    for file_name in os.listdir(path):
        data_list = {}
        if file_name.startswith("acc"):
            voice_number = file_name.replace(".txt", "").replace("acc_1_", "").replace("gyr_1_", "").split("_")[0]
            data_list[file_name] = voice_number

    return data_list


def deal_with_load_signal(path='./files_0_4/files'):
    acc_data_list = load_acc_data_with_label(path)
    for key in acc_data_list:
        acc_path = acc_data_list[key]
        gyr_path = acc_path.replace("acc", "gyr")

        print(acc_path, " ", gyr_path)


if __name__ == '__main__':
    deal_with_load_signal()

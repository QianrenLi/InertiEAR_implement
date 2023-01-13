import read_data as rd
import torch
import numpy as np

import matplotlib.pyplot as plt
def segmentation_check(reference_points,acc_t_idx):
    _i = 0
    counter = 0
    for i in range(int(len(acc_t_idx)/2)):
        if _i == len(reference_points):
            break
        if acc_t_idx[2*i] <= reference_points[_i] and acc_t_idx[(2*i)+1] >= reference_points[_i]:
            counter += 1
            _i += 1
            continue
        if acc_t_idx[2*i] > reference_points[_i]:
            _i += 1


    return 2 * counter/len(acc_t_idx)

def line_fit(acc_t):
    _x = np.arange(0,len(acc_t))
    z = np.polyfit(_x,acc_t,1)
    return z[0],z[1]
## Signal Processing Example 

# Compute Noise
noise_acc, noise_gyr = rd.noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")

## Segmentation Process Example

# acc_path = "./files_individual/test/speed_test/acc_1_0_5.txt"
# gyr_path = "./files_individual/test/speed_test/gyr_1_0_5.txt"

# acc_path = "./files_train/original_data/acc_1_2_0.txt"
# gyr_path = "./files_train/original_data/gyr_1_2_0.txt"

# acc_path = "./files_train/original_data/acc_1_3_0.txt"
# gyr_path = "./files_train/original_data/gyr_1_3_0.txt"

# acc_path = "./files_train/original_data/acc_1_4_0.txt"
# gyr_path = "./files_train/original_data/gyr_1_4_0.txt"

# acc_path = "./files_0_4/files/acc_1_0_30.txt"
# gyr_path = "./files_0_4/files/gyr_1_0_30.txt"

acc_path = "file_test/final_test/acc_1_666_10.txt"
gyr_path = "file_test/final_test/gyr_1_666_10.txt"

# Auto factor 0.1358695652173913
# acc_path = "file_test/final_test/acc_1_666_10.txt"
# gyr_path = "file_test/final_test/gyr_1_666_10.txt"

# Auto factor 0.3155339805825243
# acc_path = "file_test/speed_test/acc_1300_0.txt"
# gyr_path = "file_test/speed_test/gyr_1300_0.txt"

# Auto factor 0.29611
# acc_path = "file_test/speed_test/acc_1500_0.txt"
# gyr_path = "file_test/speed_test/gyr_1500_0.txt"

# Auto factor 0.0221
# acc_path = "file_test/walk_speed/acc_1300_201_0.txt"
# gyr_path = "file_test/walk_speed/gyr_1300_201_0.txt"


# nonlinear 1000, 0.0666
# acc_path = "file_test/amplitude_test/acc_1300_20_201_0.txt"
# gyr_path = "file_test/amplitude_test/gyr_1300_20_201_0.txt"


# nonlinear 1000, 0.08
# acc_path = "file_test/amplitude_test/acc_1300_40_201_0.txt"
# gyr_path = "file_test/amplitude_test/gyr_1300_40_201_0.txt"

# Auto factor,  0.1712
# acc_path = "file_test/amplitude_test/acc_1300_60_201_0.txt"
# gyr_path = "file_test/amplitude_test/gyr_1300_60_201_0.txt"


# reference_point = np.linspace( 8300, 744000, 201)
# Original Signal Display
acc_t, acc_xyz = rd.signal_read(acc_path)
gyr_t, gyr_xyz = rd.signal_read(gyr_path)
_temp = np.arange(0,len(acc_t))
w,b = line_fit(acc_t) 
acc_t = w * _temp + b

_temp = np.arange(0,len(gyr_t))
w,b = line_fit(gyr_t) 
gyr_t = w * _temp + b


# plt.subplot(2,1,1)
# plt.plot(tx * _temp + ty)
# plt.subplot(2,1,2)
# plt.plot(gyr_t)
# plt.show()

Fs = 400
acc_xyz = acc_xyz[int(Fs):int(len(acc_xyz) - Fs),:]
gyr_xyz = gyr_xyz[int(Fs):int(len(gyr_xyz) - Fs),:]
acc_t   = acc_t[int(Fs):int(len(acc_t) - Fs)]
gyr_t  = gyr_t[int(Fs):int(len(gyr_t)- Fs)]

# Remove mean value
acc_xyz = rd.remove_mean_value(acc_xyz)
gyr_xyz = rd.remove_mean_value(gyr_xyz)



##################### Segmentation Error #############
# Segmentation based on energy
h_seg = rd.segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, Fs = 400)
# For our proposed function, the segmentation is valid under high non_linear_factor
# segmentation_time,segmentation_idx = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= True,non_linear_factor= 10000,filter_type= 0,Energy_WIN = 200,Duration_WIN = 500,Expanding_Range = 0.2,is_test = True)

# print(segmentation_check(reference_point,segmentation_idx))
segmentation_time,segmentation_idx = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= True,non_linear_factor= 100000,filter_type= 0,
Energy_WIN = 300,Duration_WIN = 500,Expanding_Range = 0.2,is_test = True,is_auto_threshold = True)

# print(segmentation_check(reference_point,segmentation_idx))
##################### Segmentation Error ############
# # For paper proposed function, the segmentation is valid under small non_linear_factor


# acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)
seg_signal = rd.pre_processing(acc_xyz, gyr_xyz, segmentation_idx, segmentation_idx, acc_t, gyr_t,noise_acc,noise_gyr,fs = 400)
print(len(seg_signal))


# segmentation_time,segmentation_idx = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= True,non_linear_factor= 10000,filter_type= 0,Energy_WIN = 200,Duration_WIN = 500,Expanding_Range = 0.2,is_test = True)




# import matplotlib.pyplot as plt
# for i in range(len(seg_signal)):
#     plt.plot(seg_signal[i])
#     plt.show()
# print(acc_t_idx)

# Result Validation
from SENet import SENet
from data_loader import generate_signal, convert_to_spec, pad_trunc, get_silence_noise

myModel = torch.load("./model/se_type_2_net.pth", map_location=torch.device('cpu'))
# myModel = torch.load("./model/se_type_0_net.pth")
# map_location=torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
myModel = myModel.to(device)
## Inference
max_len = 900

correct_prediction = 0
total_prediction = 0

test = [6,6,7,7,8,8,9,9]

with torch.no_grad():
    i = 0
    for inputs_np in seg_signal:
        # plt.plot(inputs_np)
        # plt.show()
        
        inputs_np = pad_trunc(inputs_np, max_len)
        
        inputs = convert_to_spec(inputs_np)
        inputs = inputs.unsqueeze(0)
        # Get the input features and target labels, and put them on the GPU

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        inputs = inputs.to(device)
        # Get predictions
        outputs = myModel(inputs)

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs, 1)
        
        labels_np = 0
        prediction_np = prediction.cpu().numpy()
        print(prediction_np)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == torch.tensor([test[i]]).to(device)).sum().item()
        total_prediction += prediction.shape[0]
        i += 1


print(correct_prediction/total_prediction)

# print(len(seg_signal))
# for i in range(len(seg_signal)):
#     import matplotlib.pyplot as plt
#     plt.plot(seg_signal[i])
#     plt.show()
# Segmentation based on frequency 



# Signal Preprocessing Example





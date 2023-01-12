import read_data as rd
import torch
import numpy as np

def compute_reference_points(duration_type = 0,start_point = 7500,stop_point = 48000, ofs = 2000):
    
    if duration_type == 0:
        duration = 1.5 * ofs
    elif duration_type == 1:
        duration = 1.3 * ofs
    else:
        duration = 1.1 * ofs
        
    # reference_point = 
    pass
## Signal Processing Example 

# Compute Noise
noise_acc, noise_gyr = rd.noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")

## Segmentation Process Example
# acc_path = "./files_individual/files_0_1/acc_1_0_14.txt"
# gyr_path = "./files_individual/files_0_1/gyr_1_0_14.txt"


# acc_path = "./files_individual/test/speed_test/acc_1_0_5.txt"
# gyr_path = "./files_individual/test/speed_test/gyr_1_0_5.txt"

# acc_path = "./files_individual/test/speed_test/acc_slow_200_0.txt"
# gyr_path = "./files_individual/test/speed_test/gyr_slow_200_0.txt"

# acc_path = "./files_individual/test/speed_test/acc_medium_200_0.txt"
# gyr_path = "./files_individual/test/speed_test/gyr_medium_200_0.txt"

# acc_path = "./files_individual/test/speed_test/acc_fast_200_0.txt"
# gyr_path = "./files_individual/test/speed_test/gyr_fast_200_0.txt"

# acc_path = "./file_test/speed_test/acc_1_500_500.txt"
# gyr_path = "./file_test/speed_test/gyr_1_500_500.txt"

# acc_path = "./file_test/speed_test/acc_1_501_501.txt"
# gyr_path = "./file_test/speed_test/gyr_1_501_501.txt"

acc_path = "./file_test/speed_test/acc_1_502_502.txt"
gyr_path = "./file_test/speed_test/gyr_1_502_502.txt"



# Original Signal Display
acc_t, acc_xyz = rd.signal_read(acc_path)
gyr_t, gyr_xyz = rd.signal_read(gyr_path)

# Remove mean value
acc_xyz = rd.remove_mean_value(acc_xyz)
gyr_xyz = rd.remove_mean_value(gyr_xyz)

# Segmentation based on energy
h_seg = rd.segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, Fs = 400)

# For our proposed function, the segmentation is valid under high non_linear_factor
segmentation_time = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= True,non_linear_factor= 10000,filter_type= 0,Energy_WIN = 400,Duration_WIN = 500,Expanding_Range = 0.2)




# # For paper proposed function, the segmentation is valid under small non_linear_factor
# segmentation_time = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= True,non_linear_factor= 50,filter_type= 1)

acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)

seg_signal = rd.pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t,noise_acc,noise_gyr)
print(len(seg_signal))
# import matplotlib.pyplot as plt
# for i in range(len(seg_signal)):
#     plt.plot(seg_signal[i])
#     plt.show()
# # print(acc_t_idx)

# # Result Validation
# from SENet import SENet
# from data_loader import generate_signal, convert_to_spec, pad_trunc, get_silence_noise
# myModel = SENet()
# myModel = torch.load("./model_good/new/se_net_66.pth",map_location=torch.device('cpu'))
# # map_location=torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# myModel = myModel.to(device)
# ## Inference
# max_len = 800

# correct_prediction = 0
# total_prediction = 0

# with torch.no_grad():
#     for inputs_np in seg_signal:

        
#         inputs_np = pad_trunc(inputs_np, max_len)
        
#         inputs = convert_to_spec(inputs_np)
#         inputs = inputs.unsqueeze(0)
#         # Get the input features and target labels, and put them on the GPU

#         # Normalize the inputs
#         inputs_m, inputs_s = inputs.mean(), inputs.std()
#         inputs = (inputs - inputs_m) / inputs_s

#         # Get predictions
#         outputs = myModel(inputs)

#         # Get the predicted class with the highest score
#         _, prediction = torch.max(outputs, 1)
        
#         labels_np = 0
#         prediction_np = prediction.cpu().numpy()
#         print(prediction_np)
#         # Count of predictions that matched the target label
#         correct_prediction += (prediction == torch.tensor([0])).sum().item()
#         total_prediction += prediction.shape[0]


# print(correct_prediction/total_prediction)

# print(len(seg_signal))
# for i in range(len(seg_signal)):
#     import matplotlib.pyplot as plt
#     plt.plot(seg_signal[i])
#     plt.show()
# Segmentation based on frequency 



## Signal Preprocessing Example





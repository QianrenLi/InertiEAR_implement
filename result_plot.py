import read_data as rd
## Signal Processing Example 

# Compute Noise
noise_acc, noise_gyr = rd.noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")

## Segmentation Process Example

acc_path = "./files_individual/test/speed_test/acc_1_0_5.txt"
gyr_path = "./files_individual/test/speed_test/gyr_1_0_5.txt"

# Original Signal Display
acc_t, acc_xyz = rd.signal_read(acc_path)
gyr_t, gyr_xyz = rd.signal_read(gyr_path)

# Remove mean value
acc_xyz = rd.remove_mean_value(acc_xyz)
gyr_xyz = rd.remove_mean_value(gyr_xyz)

# Segmentation based on energy
h_seg = rd.segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, Fs = 400)

segmentation_time = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= True,non_linear_factor= 50,filter_type= 0)

acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)
# print(acc_t_idx)
seg_signal = rd.pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t,noise_acc,noise_gyr)

for i in range(len(seg_signal)):
    import matplotlib.pyplot as plt
    plt.plot(seg_signal[i])
    plt.show()
# Segmentation based on frequency 



## Signal Preprocessing Example





import read_data as rd
## Signal Processing Example 

# Compute Noise
noise_acc, noise_gyr = rd.noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")

## Segmentation Process Example

acc_path = ""
gyr_path = ""

# Original Signal Display
acc_t, acc_xyz = rd.signal_read(acc_path)
gyr_t, gyr_xyz = rd.signal_read(gyr_path)

# Remove mean value
acc_xyz = rd.remove_mean_value(acc_xyz)
gyr_xyz = rd.remove_mean_value(gyr_xyz)

# Segmentation based on energy
h_seg = rd.segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, Fs = 400)

segmentation_time = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr)
 

# Segmentation based on frequency 



## Signal Preprocessing Example





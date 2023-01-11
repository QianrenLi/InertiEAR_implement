import math, random
import torch
import torchaudio
from torchaudio import transforms
import numpy as np
import scipy

# from IPython.display import Audio
import load_data


class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=128,win_length= 64, hop_len=None):
        sig, sr = aud
        top_db = 70

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc 
        # spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig) 
        spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len, power=2,win_length=win_length)(sig)
        # Convert to decibels 
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


def pre_processing_example(isMel_spec=True):
    aud = AudioUtil.open('0_01_0.wav')
    sampling_rate = aud[1]
    # print(aud[1])
    sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024)
    aud_numpy = aud[0]
    aud_numpy = aud_numpy[0, :].numpy()

    sgram_numpy = sgram.numpy()

    spec_shape = sgram_numpy.shape
    print(spec_shape)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    time_duration = len(aud_numpy) / sampling_rate

    spec_hight = spec_shape[1]
    spec_width = spec_shape[2]
    xticks_number = 3
    yticks_number = 5

    if isMel_spec:
        xo_ticks = np.linspace(0, spec_width, xticks_number)
        xticks = np.linspace(0, time_duration, xticks_number)
        xticks_str = ['%.2f' % i for i in xticks]
        yo_ticks = np.linspace(0, spec_hight, yticks_number)
        log_sampling_rate = np.log10(sampling_rate / 2 / 700 + 1) * 2595
        yticks = np.linspace(0, log_sampling_rate, yticks_number)
        yticks = (np.power(10, yticks / 2595) - 1) * 700
        yticks_str = ['%.0f' % i for i in yticks]
        title_text = "Melspectrum of signal"
    else:

        log_sampling_rate = np.log10(sampling_rate / 2 / 700 + 1) * 2595
        yticks = np.linspace(0, log_sampling_rate, yticks_number)
        yticks = (np.power(10, yticks / 2595) - 1) * 700
        yticks_str = ['%.0f' % i for i in yticks]
        yo_ticks = yticks / yticks[-1] * spec_hight

        xo_ticks = np.linspace(0, spec_width, xticks_number)
        xticks = np.linspace(0, time_duration, xticks_number)
        xticks_str = ['%.2f' % i for i in xticks]
        title_text = "Spectrum of signal"
    # result = 2595 * np.log10((yo_ticks / spec_hight * sampling_rate / 2)/700 + 1)
    # print(result/result[-1]*spec_hight)

    fig = plt.figure()
    plt.plot(np.arange(len(aud_numpy)) / sampling_rate, aud_numpy)
    plt.xlabel("Time(s)")
    plt.ylabel("Intensity")
    plt.title("Typical Signal - \"Zero\"")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(sgram_numpy.transpose(1, 2, 0))

    plt.xticks(xo_ticks, xticks_str)
    plt.yticks(yo_ticks, yticks_str)
    plt.title(title_text)
    plt.xlabel("Time(s)")
    plt.ylabel("Frequency(Hz)")
    plt.show()


def signal_read(PATH):
    with open(PATH, 'r') as f:
        acc_content = f.read()

    import re
    xyz_patt = re.compile(r'-?[0-9]\d*\.\d+|\d+')
    txyz = xyz_patt.findall(acc_content)
    # print(type(txyz))
    # print(len(txyz))
    import numpy as np
    txyz = np.asarray(txyz)
    txyz = txyz.reshape(-1, 4)
    time = txyz[:, 0].astype(np.int64)
    position = txyz[:, 1:4].astype(np.float64)
    return time, position


def dimension_reduction(xyz):
    signal_shape = xyz.shape
    s = np.zeros((signal_shape[0]))
    for i in range(signal_shape[0]):
        _j = np.argmax(np.abs([xyz[i, 0], xyz[i, 1], xyz[i, 2]]))
        _sign = np.sign(xyz[i, _j])
        s[i] = _sign * np.linalg.norm(xyz[i, :], ord=2)
    return s


def normalization(data, ntype):
    _range = np.max(data) - np.min(data)
    if ntype == 0:
        return (data - np.min(data)) / _range
    else:
        DC_component = np.mean(data)
        data = data - DC_component
        _max = np.max(np.abs(data))
        return (data / _max)


def remove_mean_value(xyz_signal):
    for i in range(3):
        DC_component = np.mean(xyz_signal[:, i])
        xyz_signal[:, i] = xyz_signal[:, i] - DC_component

    return xyz_signal


def pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t,acc_noise, gyr_noise):
    '''
  acc_t_idx : (,2) format
  '''
    acc_s = []
    gyr_s = []
    acc_t_idx = acc_t_idx.astype('int')
    gyr_t_idx = gyr_t_idx.astype('int')

    for i in range(3):
        acc_xyz[:, i] = scipy.signal.wiener(acc_xyz[:, i] ,noise=acc_noise[i])
        gyr_xyz[:, i] = scipy.signal.wiener(gyr_xyz[:, i] ,noise=gyr_noise[i])


    for i in range(len(acc_t_idx)):
        acc_s.append(normalization(dimension_reduction(acc_xyz[acc_t_idx[i, 0]:acc_t_idx[i, 1], :]), 0))
        gyr_s.append(normalization(dimension_reduction(gyr_xyz[gyr_t_idx[i, 0]:gyr_t_idx[i, 1], :]), 0))

    signal = []
    for i in range(len(acc_s)):
        _, t_s = concate_time(acc_t[acc_t_idx[i, 0]:acc_t_idx[i, 1]], acc_s[i], gyr_t[gyr_t_idx[i, 0]:gyr_t_idx[i, 1]],
                              gyr_s[i])
        # print(t_s)
        signal.append(t_s)

    return signal


def concate_time(acc_t, acc_s, gyr_t, gyr_s):
    base_time_stamp = min(np.min(acc_t), np.min(gyr_t))
    acc_t = acc_t - base_time_stamp
    gyr_t = gyr_t - base_time_stamp
    _s = np.concatenate((acc_s, gyr_s))
    _t = np.concatenate((acc_t, gyr_t))
    _idx = np.argsort(_t)
    # print(_idx)
    t = _t[_idx]
    s = _s[_idx]

    return t, s


def signal_filter(data, fs, fstop, btype):
    '''
    btype = 'highpass', 'lowpass'
    '''
    from scipy import signal
    sos = signal.iirfilter(12, fstop, btype=btype, analog=False, fs=fs, output='sos', ftype="bessel")
    filterd = signal.sosfilt(sos, data)
    return filterd


def noise_computation(acc_PATH, gyr_PATH):
    # acc_PATH = "./data/accsilence.txt"
    # gyr_PATH = "./data/gyrsilence.txt"

    acc_t, acc_xyz = signal_read(acc_PATH)
    gyr_t, gyr_xyz = signal_read(gyr_PATH)

    # energy_acc =  np.linalg.norm(acc_xyz,axis=0,ord = 2)
    # energy_gyr = np.linalg.norm(gyr_xyz,axis=0,ord = 2)

    # energy_acc = np.power(energy_acc,2)
    # energy_gyr = np.power(energy_gyr,2)
    energy_acc = np.var(acc_xyz, axis=0)
    energy_gyr = np.var(gyr_xyz, axis=0)
    return energy_acc, energy_gyr


def energy_calculation(input_signal, window_size):
    power_signal_width = len(input_signal)
    power_signal = np.zeros(power_signal_width)
    input_signal = np.pad(input_signal, (window_size,), constant_values=(0, 0))
    for i in range(power_signal_width):
        power_signal[i] = np.sum(np.power(input_signal[i:i + window_size], 2))
    return power_signal


# Do convolution

def otus_implementation(Fs, energy_signal):
    maximum_value = np.max(energy_signal) / 3
    precision = maximum_value / Fs
    histgram = np.zeros(Fs + 1)
    # Contruct histgram
    for i in range(len(energy_signal)):
        t_val = energy_signal[i]
        if t_val > maximum_value:
            t_val = maximum_value
        idx = int(np.floor(t_val / precision))
        histgram[idx] += 1

    # normalize histgram
    histgram = histgram / np.sum(histgram)
    weighted_hist = np.multiply(histgram, np.linspace(0, maximum_value, num=Fs + 1, endpoint=True))
    global_val = sum(weighted_hist)
    cum_hist = np.cumsum(histgram)
    cum_weighted_hist = np.cumsum(weighted_hist)

    variance_hist = np.divide(np.power(cum_hist * global_val - cum_weighted_hist, 2), cum_hist * (1 - cum_hist))

    return np.argmax(variance_hist) * precision


def segmentation_correct(signal, threshold, duration_threshold, window_size, extend_region):
    index = signal < threshold
    diff_idx = np.diff(index.astype(float))
    cross_idx = np.nonzero(diff_idx)
    cross_idx = cross_idx[0]

    # Pad to even number
    if len(cross_idx) % 2 != 0:
        # pad front
        if diff_idx[cross_idx[0]] > 0:
            cross_idx = np.insert(cross_idx, 0, 0)
        # pad back
        else:
            cross_idx = np.insert(cross_idx, len(cross_idx), len(signal) - 1)
            
    
    segmented_idx = np.array(cross_idx)

    # remove peak with hard threshold and return the detection works or not
    if len(segmented_idx) > 2:
        for i in range(int(len(cross_idx) / 2)):
            if (cross_idx[2 * i+1] - cross_idx[2 * i]) < duration_threshold or np.mean(
                    signal[cross_idx[2 * i + 1]:cross_idx[2 * i + 2]]) > 0.8 * threshold:
                    segmented_idx = np.delete(segmented_idx,[2 * i,2*i+1] )
                    print(segmented_idx)
            
        for i in range(int(len(segmented_idx) / 2)):
            if segmented_idx[2 * i] - extend_region > 0:
                segmented_idx[2 * i] = segmented_idx[2 * i] - extend_region
            if segmented_idx[2 * i + 1] + extend_region < len(signal) - 1:
                segmented_idx[2 * i + 1] = segmented_idx[2 * i + 1] + extend_region
                
    
    for i in range(len(segmented_idx)):
        if segmented_idx[i] - 0.5 * window_size > 0 and segmented_idx[i] != len(signal) - 1:
            segmented_idx[i] = segmented_idx[i] - 0.5 * window_size
            
    # print(segmented_idx)
    return segmented_idx

    # print(np.nonzero(diff_idx))


class segmentation_handle():
    def __init__(self, acc_xyz, gyr_xyz, acc_t, gyr_t, Fs) -> None:
        self.acc_xyz = acc_xyz
        self.gyr_xyz = gyr_xyz
        self.acc_t = acc_t
        self.gyr_t = gyr_t
        self.Fs = Fs

    def time_stamp_alignment(self, acc_s, gyr_s, oFs):
        acc_t = self.acc_t
        gyr_t = self.gyr_t
        Fs = self.Fs

        base_time_stamp = min(np.min(acc_t), np.min(gyr_t))
        acc_t = acc_t - base_time_stamp
        gyr_t = gyr_t - base_time_stamp
        acc_t = np.linspace(np.min(acc_t),np.max(acc_t),len(acc_t))
        gyr_t = np.linspace(np.min(gyr_t),np.max(gyr_t),len(gyr_t))
        acc_s = acc_s.flatten()
        gyr_s = gyr_s.flatten()
        
        time_stamp_end = min(np.max(acc_t), np.max(gyr_t))
        # Length of the signal
        factor = int(np.ceil(oFs / Fs))
        if len(acc_t) == len(gyr_t):
            intp_length = factor * len(acc_t)
        else:
            intp_length = factor * int(np.ceil((len(acc_t) + len(gyr_t)) / 2))
        # The interpolation takes the 0 to final value
        acc_t_intp = np.linspace(max(np.min(acc_t), np.min(gyr_t)), time_stamp_end, intp_length)
        gyr_t_intp = np.linspace(max(np.min(acc_t), np.min(gyr_t)), time_stamp_end, intp_length)
        
        acc_intp_handle = scipy.interpolate.interp1d(acc_t,acc_s)
        gyr_intp_handle = scipy.interpolate.interp1d(gyr_t,gyr_s)
        
        acc_s_intp = acc_intp_handle(acc_t_intp)
        gyr_s_intp = gyr_intp_handle(gyr_t_intp)

        return acc_t_intp, acc_s_intp, gyr_t_intp, gyr_s_intp

    def segmentation(self, oFs, noise_acc, noise_gyr):
        # Need to select axix with most energy

        energy_acc = np.linalg.norm(self.acc_xyz, axis=0, ord=2)
        energy_gyr = np.linalg.norm(self.gyr_xyz, axis=0, ord=2)

        acc_s = self.acc_xyz[:, np.argmax(energy_acc)]
        gyr_s = self.gyr_xyz[:, np.argmax(energy_gyr)]

        acc_s_f = scipy.signal.wiener(acc_s, noise=None)
        gyr_s_f = scipy.signal.wiener(gyr_s, noise=None)
        

        acc_t_intp, acc_s_intp, gyr_t_intp, gyr_s_intp = self.time_stamp_alignment(acc_s_f, gyr_s_f, 400)
        # multiplied_signal = acc_s_intp * np.log(100*normalization(gyr_s_intp,0) + 1)
        # multiplied_signal = (acc_s_intp  +  gyr_s_intp) * (acc_s_intp  +  gyr_s_intp)
        multiplied_signal = gyr_s_intp * acc_s_intp 
        
        # import matplotlib.pyplot as plt
        # plt.subplot(3,1,1)
        # plt.plot(acc_s_intp)
        # plt.subplot(3,1,2)
        # plt.plot(gyr_s_intp)

        # Image Dispaly
        # f,t,zxx = scipy.signal.stft(multiplied_signal,fs = 2000)
        # plt.pcolormesh(t,f,np.log(np.abs(zxx)))
        # plt.show()

        # signal preprocessing
        multiplied_signal_f = signal_filter(multiplied_signal, fs=oFs, fstop=30, btype='highpass')
        multiplied_signal_f = scipy.signal.hilbert(multiplied_signal_f)
        power_signal = energy_calculation(np.abs(multiplied_signal_f), 50)
        power_signal = 1000 * normalization(power_signal, 0)
        # Otus thresholding
        threshold = otus_implementation(1000, np.log(power_signal + 1))
        # print(threshold)
        # multiplied_signal_f = signal_filter(abs(multiplied_signal),fs=oFs, fstop= 20, btype='lowpass')
        # threshold = otus_implementation(1000, multiplied_signal_f)

        # Correction segmentation
        segmentation_idx = segmentation_correct(np.log(power_signal + 1), threshold, 50, 50, 0.05 * oFs)
        # print(segmentation_idx)
        segmentation_time = acc_t_intp[segmentation_idx]

        # Paper filtering
        # power_signal = signal_filter(multiplied_signal,fs=oFs,fstop= 20, btype='lowpass')

        # Image Dispaly
        # import matplotlib.pyplot as plt
        # plt.subplot(3,1,3)
        # plt.plot(abs(np.log(normalization(multiplied_signal_f,0) + 1)))
        # plt.plot(abs(np.log(100 * normalization(multiplied_signal_f,0) + 1)))
        # plt.plot(multiplied_signal_f)
        # plt.plot(np.log(power_signal + 1))
        # print(np.correlate(acc_s_intp,gyr_s_intp,mode = "full"))
        # plt.plot(np.correlate(acc_s_intp,gyr_s_intp,mode = "full"))
        # plt.show()
        return segmentation_time

    def time2index(self, segmentation_time):
        acc_t = self.acc_t
        gyr_t = self.gyr_t

        base_time_stamp = min(np.min(acc_t), np.min(gyr_t))
        acc_t = acc_t - base_time_stamp
        gyr_t = gyr_t - base_time_stamp
        idx_length = len(segmentation_time)
        # print(segmentation_time)
        acc_t_idx = np.zeros(idx_length)
        gyr_t_idx = np.array(acc_t_idx)
        idx = 0
        for j in range(len(acc_t)):
            if acc_t[j] >= segmentation_time[idx]:
                acc_t_idx[idx] = j
                idx += 1
            if idx == idx_length:
                break

        idx = 0
        for j in range(len(gyr_t)):
            if gyr_t[j] >= segmentation_time[idx]:
                gyr_t_idx[idx] = j
                idx += 1
            if idx == idx_length:
                break
        # Incase a zero at end happen
        if acc_t[-1] == 0:
            acc_t_idx[-1] = len(acc_t) - 1
        if gyr_t[-1] == 0:
            gyr_t_idx[-1] = len(gyr_t) - 1

        return np.reshape(acc_t_idx, (-1, 2)), np.reshape(gyr_t_idx, (-1, 2))


def read_data_from_path(path):
    valid_data_list = {}
    noise_acc, noise_gyr = noise_computation("./data_test/acc_1_999_999.txt", "./data_test/gyr_1_999_999.txt")

    data_list = load_data.load_data_form_path(path)
    counter = 0
    t_counter = 0
    # acc_xyz  = np.array([]) 
    # gyr_xyz  = np.array([])
    # acc_t =   np.array([])
    # gyr_t = np.array([])
    for key in data_list:
        # print(data_list[key])
        # try:
        counter = counter + 1
        acc_path = data_list[key][0]
        gyr_path = data_list[key][1]
        # acc_path =  "./files_0_1/acc_1_1_101.txt"
        # gyr_path =  "./files_0_1/gyr_1_1_101.txt"
        
        acc_t, acc_xyz = signal_read(acc_path)
        gyr_t, gyr_xyz = signal_read(gyr_path)
        
        # if counter == 1:
        #     acc_xyz = _acc_xyz
        #     gyr_xyz = _gyr_xyz
        #     acc_t = _acc_t
        #     gyr_t = _gyr_t
        #     continue
        # if counter < 5:
        #     acc_xyz = np.append(acc_xyz,_acc_xyz,axis= 0)
        #     gyr_xyz = np.append(gyr_xyz,_gyr_xyz,axis= 0)
        #     acc_t = np.append(acc_t,_acc_t,axis= 0)
        #     gyr_t = np.append(gyr_t,_gyr_t,axis= 0)
        #     continue
        # # print(acc_xyz.shape)
        # counter = 0
        # t_counter = t_counter + 1
        # if t_counter < 3:
        #     continue
        
        acc_xyz = remove_mean_value(acc_xyz)
        gyr_xyz = remove_mean_value(gyr_xyz)

        h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, 400)

        segmentation_time = h_seg.segmentation(2000, noise_acc, noise_gyr)

        acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)
        # print(acc_t_idx)
        signal = pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t)
        for i in range(len(signal)):
            import matplotlib.pyplot as plt
            plt.plot(signal[i])
            plt.show()
        
        # if len(signal) == 5:
        #     valid_data_list[key] = signal
        # except:
        #     print("error_data: ", key)

    return valid_data_list


if __name__ == "__main__":
    path = "./files_0_1/"
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # path = "./data_test/"
    read_data_from_path(path)

    # noise_acc, noise_gyr = noise_computation("./files_0_4/files/acc_1_999_999.txt", "./files_0_4/files/gyr_1_999_999.txt")
    #
    # # Open Acc and Gyro
    # # acc_PATH = "./data/acczero4.txt"
    # # gyr_PATH = "./data/gyrzero4.txt"
    #
    # # Open HW path
    # acc_PATH = "./files_0_4/files/acc_1_0_20.txt"
    # gyr_PATH = "./files_0_4/files/gyr_1_0_20.txt"
    #
    # acc_t, acc_xyz = signal_read(acc_PATH)
    # gyr_t, gyr_xyz = signal_read(gyr_PATH)
    #
    # acc_xyz = remove_mean_value(acc_xyz)
    # gyr_xyz = remove_mean_value(gyr_xyz)
    #
    # # For each segmentation segments, reinitialize this
    # h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, 400)
    #
    # # Signal space reduction
    # # import numpy as np
    #
    # segmentation_time = h_seg.segmentation(2000, noise_acc, noise_gyr)
    # acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)
    #
    # # print(np.argmax(np.abs([-1,2,-5])))
    # # import matplotlib.pyplot as plt
    # # plt.figure
    # # plt.plot(gyr_s_intp)
    # # plt.show()
    #
    # signal = pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t)
    #
    # if len(signal) != 5:
    #     print("Error, signal length is not 5 ", len(signal))
    #
    # # max_len = 0
    # # average_len = 0
    # # for i in range(3):
    # #   import matplotlib.pyplot as plt
    # #   plt.figure
    # #   plt.plot(acc_xyz[:,i])
    # #   # generate spectrogram for signal[i]
    # #   # transform = torchaudio.transforms.Spectrogram(n_fft=400)
    # #   # spec = transform(torch.tensor(signal[i]))
    # #   # plt.specgram(signal[i],Fs=400)
    # #   plt.show()
    #
    # # import scipy
    # # # TODO: Need a noise power
    # # acc_s_f = scipy.signal.wiener(acc_s,noise = None)
    # # gyr_s_f = scipy.signal.wiener(gyr_s,noise = None)
    #
    # # Wiener filterig reuslt display
    # # import matplotlib.pyplot as plt
    # # plt.subplot(2,2,1)
    # # plt.plot(acc_s)
    # # plt.subplot(2,2,2)
    # # plt.plot(acc_s_f)
    # # plt.subplot(2,2,3)
    # # plt.plot(gyr_s)
    # # plt.subplot(2,2,4)
    # # plt.plot(gyr_s_f)
    # # plt.show()
    #
    # # Nomalize
    # # time,signal = concate_time(acc_t,acc_s,gyr_t,gyr_s)
    # # import matplotlib.pyplot as plt
    #
    # # plt.plot(signal)
    # # plt.plot(gyr_t)
    # # plt.show()
    #
    # # pre_processing_example(isMel_spec= False)
    # # print(type(sgram.numpy()))
    # # print(sgram)

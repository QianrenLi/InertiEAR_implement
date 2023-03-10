import math, random
import os

import torch
import torchaudio
from torchaudio import transforms
import numpy as np
import scipy
from scipy import signal

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

def line_fit(acc_t):
    _x = np.arange(0,len(acc_t))
    z = np.polyfit(_x,acc_t,1)
    return z[0],z[1]

def pre_processing_example(isMel_spec=True):
    aud = AudioUtil.open('0_01_0.wav')
    sampling_rate = aud[1]
    # print(aud[1])
    sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024)
    aud_numpy = aud[0]
    aud_numpy = aud_numpy[0, :].numpy()

    sgram_numpy = sgram.numpy()

    spec_shape = sgram_numpy.shape
    # print(spec_shape)
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


def PSNR(input_signal):
    variance = np.var(input_signal)
    mean_value = np.mean(input_signal)
    MSE_value = np.power(mean_value,2) + variance
    return np.max(np.abs(input_signal))/MSE_value

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

def down_sampling(input_signal,fs,ofs):
    width = len(input_signal)
    factor = fs / ofs
    output_signal = np.add.reduceat(input_signal, np.arange(0, width, factor))
    return output_signal

def dimension_reduction(xyz):
    signal_shape = xyz.shape
    s = np.zeros((signal_shape[0]))
    for i in range(signal_shape[0]):
        _j = np.argmax(np.abs([xyz[i, 0], xyz[i, 1], xyz[i, 2]]))
        _sign = np.sign(xyz[i, _j])
        s[i] = _sign * np.linalg.norm(xyz[i, :], ord=2)
    return s


def normalization(data, ntype):
    if len(data) > 0:
        _range = np.max(data) - np.min(data)
        if _range != 0:
            if ntype == 0:
                return (data - np.min(data)) / _range
            else:
                DC_component = np.mean(data)
                data = data - DC_component
                _max = np.max(np.abs(data))
                return (data / _max)
        return data
    else:
        return data

def remove_mean_value(xyz_signal):
    for i in range(3):
        DC_component = np.mean(xyz_signal[:, i])
        xyz_signal[:, i] = xyz_signal[:, i] - DC_component

    return xyz_signal

def high_frequency_suppression(clean_sig,fs):
    '''
    fs: sampling frequency (after doubling )
    '''
    clean_sig = signal_filter(clean_sig, fs= fs, fstop=80, btype='highpass')
    return clean_sig

def pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t,acc_noise, gyr_noise, fs = 800):
    '''
    acc_t_idx : (,2) format
    '''
    acc_s = []
    gyr_s = []


    acc_t_idx = acc_t_idx.astype('int')
    gyr_t_idx = gyr_t_idx.astype('int')

    for i in range(3):
        acc_xyz[:, i] = signal.wiener(acc_xyz[:, i] ,noise=acc_noise[i])
        gyr_xyz[:, i] = signal.wiener(gyr_xyz[:, i] ,noise=gyr_noise[i])

    

    for i in range(len(acc_t_idx)):
        acc_s.append(normalization(dimension_reduction(acc_xyz[acc_t_idx[i, 0]:acc_t_idx[i, 1], :]), 0))
        gyr_s.append(normalization(dimension_reduction(gyr_xyz[gyr_t_idx[i, 0]:gyr_t_idx[i, 1], :]), 0))
    
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    # fig = plt.figure()
    # ax = fig.add_subplot(311)
    # ax.plot(acc_xyz[acc_t_idx[0, 0]:acc_t_idx[0, 1],0])
    # ax.set_ylabel("x axis")
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    # ax.set_title("Three axis of Accerelator")
    # ax = fig.add_subplot(312)
    # ax.plot(acc_xyz[acc_t_idx[0, 0]:acc_t_idx[0, 1],1])
    # ax.set_ylabel("y axis")
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    # ax = fig.add_subplot(313)
    # ax.plot(acc_xyz[acc_t_idx[0, 0]:acc_t_idx[0, 1],2])
    # ax.set_ylabel("z axis")
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    # # plt.tight_layout()
    # plt.show()
    
    # plt.subplot(2,1,1)
    # plt.plot(acc_s[0])
    # plt.title("Acc Signal")
    # plt.subplot(2,1,2)
    # plt.plot(gyr_s[0])
    # plt.title("Gyr Signal")
    # plt.tight_layout()
    # plt.show()
    
    out_signal = []
    for i in range(len(acc_s)):
        try:
            _, t_s = concate_time(acc_t[acc_t_idx[i, 0]:acc_t_idx[i, 1]], acc_s[i],
                                    gyr_t[gyr_t_idx[i, 0]:gyr_t_idx[i, 1]],
                                    gyr_s[i])
            # import matplotlib.pyplot as plt
            # plt.plot(t_s)
            # plt.title("Concated Signal")
            # plt.show()
            # High frequency suppression
            t_s = high_frequency_suppression(t_s, fs)
            # t_s = signal_filter(t_s, fs= fs, fstop=20, btype='lowpass')
            # print(t_s)
            out_signal.append(t_s)
        except:
            print("concate error")

    return out_signal


def concate_time(acc_t, acc_s, gyr_t, gyr_s,type = 2):
    if type == 1:
        s = np.zeros(len(acc_s)+len(gyr_t))
        _array = acc_s if len(acc_s) > len(gyr_s) else gyr_s
        minimum_length = min(len(acc_s),len(gyr_s))
        maximum_length = max(len(acc_s),len(gyr_s))
        for i in range(minimum_length):
            s[2*i] = acc_s[i]
            s[2*i+1] = gyr_s[i]
        for i in range(minimum_length,maximum_length):
            s[i + minimum_length] = _array[i]
        return acc_t,s
    elif type == 2:
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
    else:
        return acc_t,acc_s



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
        # power_signal[i] = np.sum(np.power(input_signal[i:i + window_size], 2))
        # power_signal[i] = np.median(np.abs(input_signal[i:i + window_size]))
        power_signal[i] = np.mean(np.abs(input_signal[i:i + window_size]))
    return power_signal


# Do convolution

def otus_implementation(Fs, energy_signal):
    maximum_value = np.max(energy_signal) 
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
    histgram[histgram < 0.2/Fs] = 0
    weighted_hist = np.multiply(histgram, np.linspace(0, maximum_value, num=Fs + 1, endpoint=True))
    global_val = sum(weighted_hist)
    cum_hist = np.cumsum(histgram)
    cum_weighted_hist = np.cumsum(weighted_hist)

    variance_hist = np.zeros(cum_weighted_hist.shape)
    _cum_hist = cum_hist * (1 - cum_hist)
    _cum_hist_power = np.power(cum_hist * global_val - cum_weighted_hist, 2)
    for i in range(len(variance_hist)):
        if _cum_hist[i] != 0:
            # variance_hist[i] = np.divide(np.power(cum_hist * global_val - cum_weighted_hist, 2), cum_hist * (1 - cum_hist))
            variance_hist[i] = np.divide(_cum_hist_power[i], _cum_hist[i])
        else:
            variance_hist[i] = 0
    
    return np.argmax(variance_hist) * precision

# def ocd_detection(seg_signal):
#     _width = len(seg_signal)
#     _like_hood = np.zeros(_width)
#     for i in range(_width):
#         _like_hood[i] = 


# def bic_segmentation(seg_signal, W_MIN, W_MAX, N_SHIFT):
    

def segmentation_correct(seg_signal, threshold, duration_threshold, window_size, extend_region):
    index = seg_signal < threshold
    diff_idx = np.diff(index.astype(float))
    cross_idx = np.nonzero(diff_idx)
    cross_idx = cross_idx[0]

    # Pad to even number
    if len(cross_idx) % 2 != 0:
        # pad front
        if diff_idx[cross_idx[0]] > 0:
        if diff_idx[cross_idx[0]] > 0:
            cross_idx = np.insert(cross_idx, 0, 0)
        # pad back
        else:
            cross_idx = np.insert(cross_idx, len(cross_idx), len(seg_signal) - 1)
            
    
    segmented_idx = np.array(cross_idx)
    # print(segmented_idx)
    # print(len(segmented_idx))
    # remove peak with hard threshold and return the detection works or not
    _idx_delete = []

    if len(segmented_idx) > 2:

        # Remove valley
        _idx_delete = []
        for i in range(int(len(segmented_idx) / 2)  - 1):
            if (segmented_idx[2 * i+2] - segmented_idx[2 * i + 1]) <= duration_threshold or np.mean(
                    seg_signal[segmented_idx[2 * i +1]:segmented_idx[2 * i + 2]]) > 0.8 * threshold:
                    _idx_delete.append(2 * i + 1)
                    _idx_delete.append(2 * i + 2)
        if len(_idx_delete) > 0:
            segmented_idx = np.delete(segmented_idx,_idx_delete)

        # print(segmented_idx)

        # Remove peak
        _idx_delete = []
        for i in range(int(len(segmented_idx) / 2)):
            if (segmented_idx[2 * i+1] - segmented_idx[2 * i]) <= duration_threshold or np.mean(
                    seg_signal[segmented_idx[2 * i ]:segmented_idx[2 * i + 1]]) < 0.2 * threshold:
                    _idx_delete.append(2 * i)
                    _idx_delete.append(2 * i + 1)
        if len(_idx_delete) > 0:
            segmented_idx = np.delete(segmented_idx,_idx_delete)
        # print(segmented_idx)

        # print(segmented_idx)
    




    for i in range(int(len(segmented_idx) / 2)):
        if segmented_idx[2 * i] - extend_region > 0:
            segmented_idx[2 * i] = segmented_idx[2 * i] - extend_region
        else:
            segmented_idx[2 * i] = 0
        if segmented_idx[2 * i + 1] + extend_region < len(seg_signal) - 1:
            segmented_idx[2 * i + 1] = segmented_idx[2 * i + 1] + extend_region
        else:
            segmented_idx[2 * i + 1] = len(seg_signal) - 1
                
        
    for i in range(len(segmented_idx)):
        if segmented_idx[i] - 0.5 * window_size > 0 and segmented_idx[i] != len(seg_signal) - 1:
            segmented_idx[i] = segmented_idx[i] - 0.5 * window_size

        
        
    # print(segmented_idx)
    # print(len(segmented_idx))
    # print(segmented_idx)
    return segmented_idx
    return segmented_idx

    # print(np.nonzero(diff_idx))    

def window_energy_computation(xyz_data,window_size):
    power_signal = np.zeros(3)
    for i in range(3):
        power_signal_width = len(xyz_data[:,i])
        input_signal = np.pad(xyz_data[:,i].flatten(), (window_size,), constant_values=(0, 0))
        input_signal = median_filter(input_signal,7)
        # signal_var = 3 * np.var(input_signal)
        # print(signal_var)
        # input_signal[abs(input_signal) > signal_var] = signal_var
        # input_signal = normalization(input_signal,1)
        # import matplotlib.pyplot as plt
        # plt.plot(input_signal)
        # plt.show()
        for j in range(power_signal_width):
            # power_signal[i] = np.sum(np.power(input_signal[i:i + window_size], 2))
            power_signal[i] += np.mean(np.abs(input_signal[j:j + window_size]))
        
    return power_signal

def median_filter(input_signal,window_size):
    _input_signal = np.pad(input_signal, (window_size,), constant_values=(0, 0))
    for i in range(len(input_signal)):
        input_signal[i] = np.median(_input_signal[i:i+window_size])
    return input_signal

def spectral_entropy_calculation(xyz_data):
    power_signal = np.zeros(3)
    for i in range(3):
        input_signal = xyz_data[:,i].flatten()
        input_signal = normalization(input_signal,1)
        signal_spec = np.power(np.fft.fft(input_signal),2)
        probability_distribution = signal_spec/sum(signal_spec)
        power_signal[i] = sum(probability_distribution * np.log2(probability_distribution))/np.log2(len(input_signal))
    return power_signal

def create_hist(energy_signal,Fs):
    maximum_value = np.max(energy_signal)
    precision = maximum_value  / Fs
    histgram = np.zeros(Fs + 1)
    # Contruct histgram
    for i in range(len(energy_signal)):
        t_val = energy_signal[i]
        if t_val > maximum_value:
            t_val = maximum_value
        idx = int(np.floor(np.floor(t_val / precision)))
        histgram[idx] += 1

    # normalize histgram
    histgram = histgram / np.sum(histgram)
    return histgram

def entropy_calculation(xyz_data):
    power_signal = np.zeros(3)
    for i in range(3):
        input_signal = np.abs(xyz_data[:,i].flatten())
        
        probability_distribution = create_hist(input_signal,10000)
        power_signal[i] = sum(probability_distribution * np.log2(probability_distribution))/np.log2(len(input_signal))
    return power_signal
        

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
        acc_t_intp = np.linspace(max(np.min(acc_t), np.min(gyr_t)) , time_stamp_end, intp_length)
        gyr_t_intp = np.linspace(max(np.min(acc_t), np.min(gyr_t)), time_stamp_end, intp_length)
        
        acc_intp_handle = scipy.interpolate.PchipInterpolator(acc_t,acc_s)
        gyr_intp_handle = scipy.interpolate.PchipInterpolator(gyr_t,gyr_s)

        acc_s_intp = acc_intp_handle(acc_t_intp)
        gyr_s_intp = gyr_intp_handle(gyr_t_intp)

        return acc_t_intp, acc_s_intp, gyr_t_intp, gyr_s_intp

    def segmentation(self, oFs, noise_acc, noise_gyr, is_plot = False, non_linear_factor = 10, filter_type = 0, 
                     Energy_WIN = 200, Duration_WIN = 210, Expanding_Range = 0.3, is_test = False,is_auto_threshold = False):
        # Need to select axix with most energy
            
        for i in range(3):
            self.acc_xyz[:, i] = signal.wiener(self.acc_xyz[:, i] ,noise=noise_acc[i])
            self.gyr_xyz[:, i] = signal.wiener(self.gyr_xyz[:, i] ,noise=noise_gyr[i])
        energy_acc = window_energy_computation(self.acc_xyz, window_size = 20)
        energy_gyr = window_energy_computation(self.gyr_xyz, window_size = 20)
        # energy_acc = spectral_entropy_calculation(self.acc_xyz)
        # energy_gyr = spectral_entropy_calculation(self.gyr_xyz)
        # energy_acc = entropy_calculation(self.acc_xyz)
        # energy_gyr = entropy_calculation(self.gyr_xyz)
        # energy_acc = np.linalg.norm(self.acc_xyz, axis=0, ord=2)
        # energy_gyr = np.linalg.norm(self.gyr_xyz, axis=0, ord=2)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(311)
        # ax.plot(self.acc_xyz[:,0])
        # ax = fig.add_subplot(312)
        # ax.plot(self.acc_xyz[:,1])
        # ax.set_ylabel("y axis")
        # ax = fig.add_subplot(313)
        # ax.plot(self.acc_xyz[:,2])
        # ax.set_ylabel("z axis")
        # # plt.tight_layout()
        # plt.show()
        
        # acc_s = self.acc_xyz[:, np.argmax(energy_acc)]
        # gyr_s = self.gyr_xyz[:, np.argmax(energy_gyr)]
        acc_s = self.acc_xyz[:, np.argmax(energy_acc)]
        gyr_s = self.gyr_xyz[:, np.argmax(energy_gyr)]
        

        acc_s_f = acc_s
        gyr_s_f = gyr_s


        acc_t_intp, acc_s_intp, gyr_t_intp, gyr_s_intp = self.time_stamp_alignment(acc_s_f, gyr_s_f, oFs)

        result_signal = normalization(acc_s_intp,1) + 0.5 *  normalization(gyr_s_intp,1)
        
        multiplied_signal = result_signal * result_signal
        # multiplied_signal = gyr_s_intp * acc_s_intp 
        multiplied_signal = median_filter(multiplied_signal,7)

        # f, t, Zxx = scipy.signal.stft(multiplied_signal,fs = 400)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.log2(abs(Zxx)),origin = 'lower',aspect='auto')
        # plt.yticks(np.linspace(0,len(f)-1,5),f[np.linspace(0,len(f)-1,5).astype(np.int32)])
        # plt.xticks(np.linspace(0,len(t)-1,5),t[np.linspace(0,len(t)-1,5).astype(np.int32)])
        # plt.title("Specturum Example")
        # plt.ylabel("Frequency")
        # plt.xlabel("Time sample")
        # plt.show()

        
        if filter_type == 1:
            multiplied_signal = abs(multiplied_signal)
            multiplied_signal_f = signal_filter(multiplied_signal, fs=oFs, fstop=20, btype='lowpass')
            power_signal = np.abs(non_linear_factor *normalization(multiplied_signal_f,0))
            threshold = otus_implementation(1000, np.log(power_signal + 1))
            segmentation_idx = segmentation_correct(np.log(power_signal + 1), threshold, Energy_WIN, Duration_WIN, Expanding_Range * oFs)
            segmentation_time = acc_t_intp[segmentation_idx]
            segmentation_idx = np.reshape(segmentation_idx/(oFs/self.Fs),(-1,2))
        else:
            # signal preprocessing
            multiplied_signal_f = signal_filter(multiplied_signal, fs=oFs, fstop=100, btype='highpass')
            multiplied_signal_f = signal.hilbert(multiplied_signal_f)
            power_signal = energy_calculation(np.abs(multiplied_signal_f), Energy_WIN)
            if is_auto_threshold:
                non_linear_factor = PSNR(power_signal)/10

            power_signal = non_linear_factor * normalization(power_signal, 0)
            
            # Otus thresholding
            threshold = otus_implementation(10000, np.log(power_signal + 1))
            segmentation_idx = segmentation_correct(np.log(power_signal + 1), threshold, Energy_WIN, Duration_WIN, Expanding_Range * oFs)
            segmentation_time = acc_t_intp[segmentation_idx]
            segmentation_idx = np.reshape(segmentation_idx/(oFs/self.Fs),(-1,2))


            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,2)
            # plt.plot(np.log(power_signal + 1))
            # plt.subplot(2,1,1)
            # plt.plot(multiplied_signal)
            # plt.show()
            # Correction segmentation
        # if len(segmentation_time) != 2:
        #     print(segmentation_idx)
        if is_plot == True:
            import matplotlib.pyplot as plt
            # plt.figure(figsize=(16,8))
            plt.subplot(4,1,1)
            plt.plot(acc_s_intp)
            plt.title("Interpolated Accelerometer data")
            plt.subplot(4,1,2)
            plt.plot(gyr_s_intp)
            plt.title("Interpolated Gyroscope data")
            plt.subplot(4,1,3)
            plt.plot(multiplied_signal)
            plt.title("Multiplied Data")
            plt.subplot(4,1,4)
            _xn = np.arange(len(power_signal))
            line1, = plt.plot(_xn,np.log(power_signal + 1))
            line2, = plt.plot(_xn,threshold * np.ones(_xn.shape))
            # plt.plot(_xn,np.log(power_signal + 1),_xn,threshold * np.ones(_xn.shape),linestyle="solid")
            plt.legend(handles= [line1,line2],labels = ["Envelop","Threshold"], loc='best')
            plt.title("Envelop of data")
            plt.tight_layout()
            plt.show()

        if is_test:
            return segmentation_time,segmentation_idx

        # Paper filtering
        # power_signal = signal_filter(multiplied_signal,fs=oFs,fstop= 20, btype='lowpass')
        return segmentation_idx

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


def data_processing(acc_path,gyr_path,file_directory,label):
    noise_acc, noise_gyr = noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")
    acc_t, acc_xyz = signal_read(acc_path)
    gyr_t, gyr_xyz = signal_read(gyr_path)
    Fs = 400
    acc_xyz = acc_xyz[int(Fs):int(len(acc_xyz) - Fs),:]
    gyr_xyz = gyr_xyz[int(Fs):int(len(gyr_xyz) - Fs),:]
    acc_t   = acc_t[int(Fs):int(len(acc_t) - Fs)]
    gyr_t  = gyr_t[int(Fs):int(len(gyr_t)- Fs)]
    _temp = np.arange(0,len(acc_t))
    w,b = line_fit(acc_t) 
    acc_t = w * _temp + b

    _temp = np.arange(0,len(gyr_t))
    w,b = line_fit(gyr_t) 
    gyr_t = w * _temp + b

    acc_xyz = remove_mean_value(acc_xyz)
    gyr_xyz = remove_mean_value(gyr_xyz)
    
    h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, Fs = 400)
    
    segmentation_time,segmentation_idx =h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr,is_plot= False,non_linear_factor= 1000,filter_type= 0,
Energy_WIN = 200,Duration_WIN = 500,Expanding_Range = 0.2,is_test = True,is_auto_threshold = True)
    
    acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)
    
    seg_signal = pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t,noise_acc,noise_gyr)
    import os
    voice_number = 0
    for cur_file_name in os.listdir(file_directory):
        if cur_file_name.startswith("signal"):
            _voice_number = int(cur_file_name.replace(".npy", "").replace("signal_", "").split("_")[2])
            voice_number = max(voice_number,_voice_number)
    print(len(seg_signal))
    for i in range(len(seg_signal)):
        print("signal len: ", len(seg_signal[i]))
        np.save(("%ssignal_1_%d_%d") % (file_directory,label,i + voice_number + 1),seg_signal[i])
    
    # Use example
    # Save Example
    # data_processing(file_directory="./files_individual/files_signal/files_0/",acc_path="./files_individual/test/speed_test/acc_slow_200_0.txt",gyr_path="files_individual/test/speed_test/acc_slow_200_0.txt",label= 0)
    # Load Example
    # signal = np.load(filename)
    
def read_data_from_path(path):
    valid_data_list = {}
    noise_acc, noise_gyr = noise_computation("./files_individual/noise/acc_1_999_999.txt", "./files_individual/noise/gyr_1_999_999.txt")

    data_list = load_data.load_data_form_path(path)

    for key in data_list:
        try:
            acc_path = data_list[key][0]
            gyr_path = data_list[key][1]
            
            acc_t, acc_xyz = signal_read(acc_path)
            gyr_t, gyr_xyz = signal_read(gyr_path)
            
            
            acc_xyz = remove_mean_value(acc_xyz)
            gyr_xyz = remove_mean_value(gyr_xyz)

            h_seg = segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t, Fs = 400)

            segmentation_time = h_seg.segmentation(oFs = 2000, noise_acc = noise_acc, noise_gyr = noise_gyr)

            acc_t_idx, gyr_t_idx = h_seg.time2index(segmentation_time=segmentation_time)
            # print(acc_t_idx)
            seg_signal = pre_processing(acc_xyz, gyr_xyz, acc_t_idx, gyr_t_idx, acc_t, gyr_t,noise_acc,noise_gyr)
            # if len(seg_signal) != 1:
            #     print(acc_t_idx)
            # for i in range(len(seg_signal)):
            #     import matplotlib.pyplot as plt
            #     plt.subplot(2,1,1)
            #     plt.plot(seg_signal[i])
            #     plt.subplot(2,1,2)
            #     plt.plot(dimension_reduction(acc_xyz))
            #     plt.show()
                # print(len(seg_signal))
            if len(seg_signal) == 1:
                valid_data_list[key] = seg_signal
        except:
            print("error_data: ", key)

    return valid_data_list


if __name__ == "__main__":
    in_dir_path = "files_train/original_data_new"
    out_dir_path = "files_train/signal_data_type_1"
    for file_name in os.listdir(in_dir_path):
        if file_name.count("acc"):
            acc_file = file_name
            gyr_file = file_name.replace("acc", "gyr")
            label = int(file_name.replace(".txt", "").split("_")[-2])
            print(acc_file, gyr_file, label)
            out_dir_path_i = out_dir_path + "/files_" + str(label)
            if not os.path.isdir(out_dir_path_i):
                os.mkdir(out_dir_path_i)
            data_processing(acc_path=in_dir_path + "/" + acc_file, gyr_path=in_dir_path + "/" + gyr_file,
                            file_directory=out_dir_path_i + "/", label=label)

    # max_signal_len = 0
    # total_signal_len = 0
    # count = 0
    # for i in range(0, 10):
    #     data_dir = out_dir_path + "/files_" + str(i)
    #     for file_name in os.listdir(data_dir):
    #         signal = np.load(data_dir + "/" + file_name)
    #         cur_len = len(signal)
    #         if cur_len > max_signal_len:
    #             max_signal_len = cur_len
    #         total_signal_len += cur_len
    #         count += 1
    #         print(str(i) + ": ", cur_len, " max_signal_len: ", max_signal_len,
    #               " average_signal_len: ", total_signal_len / count, " count: ", count)



import math, random 
import torch 
import torchaudio 
from torchaudio import transforms 
import numpy as np
import scipy
# from IPython.display import Audio 
 
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
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None): 
        sig,sr = aud 
        top_db = 80 
    
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc 
        # spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig) 
        spec = transforms.Spectrogram( n_fft=n_fft, hop_length=hop_len,power = 2)(sig) 
        # Convert to decibels 
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec) 
        return (spec)

def pre_processing_example(isMel_spec = True):
    aud = AudioUtil.open('0_01_0.wav')
    sampling_rate = aud[1]
    # print(aud[1])
    sgram = AudioUtil.spectro_gram(aud,n_mels=64, n_fft=1024)
    aud_numpy = aud[0]
    aud_numpy = aud_numpy[0,:].numpy()

    
    sgram_numpy = sgram.numpy()

    spec_shape = sgram_numpy.shape
    print(spec_shape)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    time_duration = len(aud_numpy)/ sampling_rate

    spec_hight = spec_shape[1]
    spec_width = spec_shape[2]
    xticks_number = 3
    yticks_number = 5

    if isMel_spec:
        xo_ticks = np.linspace(0,spec_width,xticks_number)
        xticks = np.linspace(0,time_duration,xticks_number)
        xticks_str = ['%.2f'% i for i in xticks]
        yo_ticks = np.linspace(0,spec_hight,yticks_number)
        log_sampling_rate = np.log10(sampling_rate/2/700 + 1) * 2595
        yticks = np.linspace(0,log_sampling_rate,yticks_number)
        yticks = (np.power(10,yticks/2595) - 1) * 700
        yticks_str = ['%.0f'% i for i in yticks]
        title_text = "Melspectrum of signal"
    else:

        log_sampling_rate = np.log10(sampling_rate/2/700 + 1) * 2595
        yticks = np.linspace(0,log_sampling_rate,yticks_number)
        yticks = (np.power(10,yticks/2595) - 1) * 700
        yticks_str = ['%.0f'% i for i in yticks]
        yo_ticks = yticks/yticks[-1] * spec_hight

        xo_ticks = np.linspace(0,spec_width,xticks_number)
        xticks = np.linspace(0,time_duration,xticks_number)
        xticks_str = ['%.2f'% i for i in xticks]
        title_text = "Spectrum of signal"
    # result = 2595 * np.log10((yo_ticks / spec_hight * sampling_rate / 2)/700 + 1)
    # print(result/result[-1]*spec_hight)

    fig = plt.figure()
    plt.plot(np.arange(len(aud_numpy))/sampling_rate,aud_numpy)
    plt.xlabel("Time(s)")
    plt.ylabel("Intensity")
    plt.title("Typical Signal - \"Zero\"")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(sgram_numpy.transpose(1,2,0))


    plt.xticks(xo_ticks, xticks_str)
    plt.yticks(yo_ticks ,yticks_str)
    plt.title(title_text)
    plt.xlabel("Time(s)")
    plt.ylabel("Frequency(Hz)")
    plt.show()
    
def signal_read(PATH):
    with open(PATH,'r') as f:
      acc_content = f.read()

    import re
    xyz_patt = re.compile(r'-?[0-9]\d*\.\d+|\d+') 
    txyz = xyz_patt.findall(acc_content)
    # print(type(txyz))
    # print(len(txyz))
    import numpy as np
    txyz = np.asarray(txyz)
    txyz = txyz.reshape(-1,4)
    time = txyz[:,0].astype(np.int64)
    position = txyz[:,1:4].astype(np.float64)
    return time,position

def dimension_reduction(xyz):
    signal_shape = xyz.shape
    s = np.zeros((signal_shape[0],1))
    for i in range(signal_shape[0]):
      _j = np.argmax(np.abs([xyz[i,0],xyz[i,1],xyz[i,2]]))
      _sign = np.sign(xyz[i,_j])
      s[i] = _sign * np.linalg.norm(xyz[i,:],ord = 2)
    return s

def normalization(data, ntype):
    _range = np.max(data) - np.min(data)
    if ntype == 0:
      return (data - np.min(data)) / _range
    else:
      DC_component = np.mean(data)
      data = data - DC_component
      return (data)


def concate_time(acc_t,acc_s,gyr_t,gyr_s):
    base_time_stamp = min(np.min(acc_t),np.min(gyr_t))
    acc_t = acc_t - base_time_stamp
    gyr_t = gyr_t - base_time_stamp
    _s = np.concatenate((acc_s,gyr_s))
    _t = np.concatenate((acc_t,gyr_t))
    _idx = np.argsort(_t)
    # print(_idx)
    t = _t[_idx]
    s = _s[_idx]
    return t,s

def butter_filter(data,fs,fstop,btype):
    '''
    btype = 'highpass', 'lowpass'
    '''
    from scipy import signal
    sos = signal.butter(4,fstop,btype=btype,analog=False,fs = fs,output='sos')
    filterd = signal.sosfilt(sos,data)
    return filterd


def time_stamp_alignment(acc_t,acc_s,gyr_t,gyr_s,Fs,oFs):
    base_time_stamp = min(np.min(acc_t),np.min(gyr_t))
    acc_t = acc_t - base_time_stamp
    gyr_t = gyr_t - base_time_stamp
    gyr_s = gyr_s.flatten()
    acc_s = acc_s.flatten()
    time_stamp_end = max(np.max(acc_t),np.max(gyr_t))
    # Length of the signal
    factor = int(np.ceil(oFs / Fs))
    if len(acc_t) == len(gyr_t):
        intp_length = factor * len(acc_t)
    else:
        intp_length = factor * int(np.ceil((len(acc_t) + len(gyr_t)) / 2))
    # The interpolation takes the 0 to final value
    acc_t_intp = np.linspace(0, time_stamp_end,intp_length)
    gyr_t_intp =  np.linspace(0, time_stamp_end,intp_length)

    acc_s_intp = np.interp(acc_t_intp,acc_t,acc_s)
    gyr_s_intp = np.interp(gyr_t_intp,gyr_t,gyr_s)

    return acc_t_intp,acc_s_intp,gyr_t_intp,gyr_s_intp

def segmentation(acc_xyz,gyr_xyz,acc_t,gyr_t,Fs,oFs,thres_hold):
    # Need to select axix with most energy
    energy_acc =  np.linalg.norm(acc_xyz,axis=0,ord = 2)
    energy_gyr = np.linalg.norm(gyr_xyz,axis=0,ord = 2)
    acc_s = acc_xyz[:,np.argmax(energy_acc)]
    gyr_s = gyr_xyz[:,1]
    acc_s = normalization(acc_s,1)
    gyr_s = normalization(gyr_s,1)
    acc_s_f = scipy.signal.wiener(acc_s,noise = 0.0005)
    gyr_s_f = scipy.signal.wiener(gyr_s,noise = 0.0005)
    acc_t_intp,acc_s_intp,gyr_t_intp,gyr_s_intp = time_stamp_alignment(acc_t,acc_s_f,gyr_t,gyr_s_f,Fs,oFs)
    multiplied_signal = acc_s_intp * acc_s_intp

    multiplied_signal_f = butter_filter(multiplied_signal,fs=oFs,fstop= 10,btype='lowpass')
    import matplotlib.pyplot as plt 
    plt.plot(multiplied_signal_f)
    plt.show()
    pass
    
  
  # Do convolution
  
  
  
# Open Acc and Gyro
acc_PATH = "./data/acczero.txt"
gyr_PATH = "./data/gyrzero.txt"

acc_t,acc_xyz = signal_read(acc_PATH)
gyr_t,gyr_xyz = signal_read(gyr_PATH)


# Signal space reduction
# import numpy as np
acc_s = dimension_reduction(acc_xyz)
gyr_s = dimension_reduction(gyr_xyz)

segmentation(acc_xyz,gyr_xyz,acc_t,gyr_t,400,2000,20)


acc_t_intp,acc_s_intp,gyr_t_intp,gyr_s_intp = time_stamp_alignment(acc_t,acc_s,gyr_t,gyr_s,400,2000)


# print(np.argmax(np.abs([-1,2,-5])))
import matplotlib.pyplot as plt
plt.figure
plt.plot(gyr_s_intp)
plt.show()

acc_s = normalization(acc_s,0)
gyr_s = normalization(gyr_s,0)


import matplotlib.pyplot as plt
plt.figure
plt.plot(gyr_s)
plt.show()

# import scipy
# # TODO: Need a noise power
# acc_s_f = scipy.signal.wiener(acc_s,noise = None)
# gyr_s_f = scipy.signal.wiener(gyr_s,noise = None)




# Wiener filterig reuslt display
# import matplotlib.pyplot as plt
# plt.subplot(2,2,1)
# plt.plot(acc_s)
# plt.subplot(2,2,2)
# plt.plot(acc_s_f)
# plt.subplot(2,2,3)
# plt.plot(gyr_s)
# plt.subplot(2,2,4)
# plt.plot(gyr_s_f)
# plt.show()

# Nomalize
time,signal = concate_time(acc_t,acc_s,gyr_t,gyr_s)
import matplotlib.pyplot as plt

plt.plot(signal)
# plt.plot(gyr_t)
plt.show()




# pre_processing_example(isMel_spec= False)
# print(type(sgram.numpy()))
# print(sgram)
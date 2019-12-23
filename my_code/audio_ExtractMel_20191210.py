#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:32:28 2018

@author: hxj
"""

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import cPickle
import sys

# import base
# import sigproc
audio_file = sys.argv[1]
eps = 1e-5


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def getlogspec(signal, samplerate=16000, winlen=0.02, winstep=0.01,
               nfilt=26, nfft=399, lowfreq=0, highfreq=None, preemph=0.97,
               winfunc=lambda x: np.ones((x,))):
    highfreq = highfreq or samplerate / 2
    signal = ps.sigproc.preemphasis(signal, preemph)
    frames = ps.sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = ps.sigproc.logpowspec(frames, nfft)
    return pspec


def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    # wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def zscore(data, mean, std):
    shape = np.array(data.shape, dtype=np.int32)
    for i in range(shape[0]):
        data[i, :, :, 0] = (data[i, :, :, 0] - mean) / (std)
    return data


def normalization(data):
    '''
    #apply zscore
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    train_data = zscore(train_data,mean,std)
    test_data = zscore(test_data,mean,std)
    '''
    mean = np.mean(data, axis=0)  # axis=0纵轴方向求均值
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data


def mapminmax(data):
    shape = np.array(data.shape, dtype=np.int32)
    for i in range(shape[0]):
        min = np.min(data[i, :, :, 0])
        max = np.max(data[i, :, :, 0])
        data[i, :, :, 0] = (data[i, :, :, 0] - min) / ((max - min) + eps)
    return data


def generate_label(emotion, classnum):
    label = -1
    if (emotion == 'ang'):
        label = 0
    elif (emotion == 'sad'):
        label = 1
    elif (emotion == 'hap'):
        label = 2
    elif (emotion == 'neu'):
        label = 3
    elif (emotion == 'fear'):
        label = 4
    else:
        label = 5
    return label


def load_data():
    f = open('./audio_zscore20191210_40.pkl', 'rb')
    mean1, std1, mean2, std2, mean3, std3 = cPickle.load(f)
    return mean1, std1, mean2, std2, mean3, std3


def read_IEMOCAP():
    eps = 1e-5
    tnum = 1  # the number of test utterance
    test_num = 420  # the number of test 2s segments
    filter_num = 40
    pernums_test = np.arange(tnum)  # remerber each utterance contain how many segments


    mean1, std1, mean2, std2, mean3, std3 = load_data()


    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)

    tnum = 0
    test_num = 0

    data, time, rate = read_file(audio_file)
    mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    # apply zscore

    time = mel_spec.shape[0]
    # training set
    if (time <= 300):
        pernums_test[tnum] = 1
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                      constant_values=0)
        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant',
                         constant_values=0)
        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant',
                         constant_values=0)
        test_data[test_num, :, :, 0] = (part - mean1) / (std1 + eps)
        test_data[test_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
        test_data[test_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)
        test_num = test_num + 1
        tnum = tnum + 1
    else:
        pernums_test[tnum] = 2
        frames = divmod(time - 300, 100)[0] + 1
        for i in range(frames):
            begin = 100 * i
            end = begin + 300
            part = mel_spec[begin:end, :]
            delta11 = delta1[begin:end, :]
            delta21 = delta2[begin:end, :]
            test_data[test_num, :, :, 0] = (part - mean1) / (std1 + eps)
            test_data[test_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
            test_data[test_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)
            test_num = test_num + 1
            tnum = tnum + 1

    for m in range(1):
        arr = np.arange(test_num)
        np.random.shuffle(arr)
        output = './IEMOCAP_20191210.pkl'
        f = open(output, 'wb')
        cPickle.dump((test_data, pernums_test), f)
        f.close()
    return


if __name__ == '__main__':
    read_IEMOCAP()
    # print "test_num:", test_num
    # print "train_num:", train_num
#    n = wgn(x, 6)
#    xn = x+n # 增加了6dBz信噪比噪声的信号

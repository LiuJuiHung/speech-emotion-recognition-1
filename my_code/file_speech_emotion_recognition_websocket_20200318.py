#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:23:45 2018

@author: hxj

Update on Fri Dec 13 22:19:23 2019
@author: LiuJuiHung
"""
# ===== Recognition Emotion =====
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from acrnn1 import *

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import cPickle
import sys
import time
from os import listdir
from os.path import join
import csv
import pandas as pd
import threading
import json
# import base
# import sigproc

# audio_file = sys.argv[1]

# ===== socket to pi=====
import socket

# ===== websocket =====
import websocket
uri = "ws://localhost:8000/ws/polls/liujuihung/"
ws = None

BASE_DIR = '/media/mmnlab/6cf5f717-b8af-4061-a005-df237cd25492/home/mmnlab/PycharmProjects/speech-emotion-recognition-RNN-2/speech-emotion-recognition-1/my_code/'
list_conn = []


eps = 1e-5

def save_csv(name, data):
    with open(str({name}) + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # for rows in data:
        writer.writerow(data)

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

## zscore
def audio_zscore(audio_file):
    train_num = 2928
    filter_num = 40
    traindata1 = np.empty((train_num * 300, filter_num), dtype=np.float32)
    traindata2 = np.empty((train_num * 300, filter_num), dtype=np.float32)
    traindata3 = np.empty((train_num * 300, filter_num), dtype=np.float32)
    train_num = 0

    data, time, rate = read_file(audio_file)
    mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)

    time = mel_spec.shape[0]

    # training set
    if (time <= 300):
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                      constant_values=0)
        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant',
                         constant_values=0)
        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant',
                         constant_values=0)
        traindata1[train_num * 300:(train_num + 1) * 300] = part
        traindata2[train_num * 300:(train_num + 1) * 300] = delta11
        traindata3[train_num * 300:(train_num + 1) * 300] = delta21

        train_num = train_num + 1
    else:
        frames = divmod(time - 300, 100)[0] + 1
        for i in range(frames):
            begin = 100 * i
            end = begin + 300
            part = mel_spec[begin:end, :]
            delta11 = delta1[begin:end, :]
            delta21 = delta2[begin:end, :]
            traindata1[train_num * 300:(train_num + 1) * 300] = part
            traindata2[train_num * 300:(train_num + 1) * 300] = delta11
            traindata3[train_num * 300:(train_num + 1) * 300] = delta21
            train_num = train_num + 1

    mean1 = np.mean(traindata1, axis=0)  # axis=0纵轴方向求均值
    std1 = np.std(traindata1, axis=0)
    mean2 = np.mean(traindata2, axis=0)  # axis=0纵轴方向求均值
    std2 = np.std(traindata2, axis=0)
    mean3 = np.mean(traindata3, axis=0)  # axis=0纵轴方向求均值
    std3 = np.std(traindata3, axis=0)
    # output = './audio_zscore20191210_' + str(filter_num) + '.pkl'
    # # output = './IEMOCAP'+str(m)+'_'+str(filter_num)+'.pkl'
    # f = open(output, 'wb')
    # cPickle.dump((mean1, std1, mean2, std2, mean3, std3), f)
    # f.close()
    # save_csv('mean11', mean1)
    # save_csv('std11', std1)
    # save_csv('mean22', mean2)
    # save_csv('std22', std2)
    # save_csv('mean33', mean3)
    # save_csv('std33', std3)
    return mean1, std1, mean2, std2, mean3, std3


## audio read
def read_audio(mean1, std1, mean2, std2, mean3, std3, audio_file):
    eps = 1e-5
    tnum = 1  # the number of test utterance
    test_num = 420  # the number of test 2s segments
    filter_num = 40
    pernums_test = np.arange(tnum)  # remerber each utterance contain how many segments


    # mean1, std1, mean2, std2, mean3, std3 = load_data()


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
        # output = './IEMOCAP_20191210.pkl'
        # f = open(output, 'wb')
        # cPickle.dump((test_data, pernums_test), f)
        # f.close()
    # save_csv('test_data2', test_data)
    # save_csv('pernums_test2', pernums_test)
    return test_data, pernums_test


## Predict
tf.app.flags.DEFINE_integer('num_epoch', 5000, 'The number of epoches for training.')
tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 30, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('is_adam', True, 'whether to use adam optimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'learning rate of Adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1, 'the prob of every unit keep in dropout layer')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')
## My code
tf.app.flags.DEFINE_string('testdata_path', './IEMOCAP_20191210.pkl', 'total dataset includes training set')
tf.app.flags.DEFINE_string('checkpoint', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('model_name', 'model4.ckpt', 'model name')

FLAGS = tf.app.flags.FLAGS

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def predict(test_data, pernums_test):
    #####reset model meta graph##########
    tf.reset_default_graph()

    ## My code
    # test_data, pernums_test = load_data(FLAGS.testdata_path)
    test_size = test_data.shape[0]
    tnum = pernums_test.shape[0]

    ## Original Code for Valid
    # rain_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = load_data(FLAGS.testdata_path)
    # valid_size = valid_data.shape[0]
    # vnum = pernums_valid.shape[0]


    ##########tarin model###########
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    is_training = tf.placeholder(tf.bool)
    # lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = acrnn(X, is_training=is_training, dropout_keep_prob=keep_prob)
    # Ylogits = tf.nn.softmax(Ylogits)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, '../checkpoint30/model4.ckpt-3146')
        # Watch all model's variables
        # all_vars = tf.trainable_variables()
        # for v in all_vars:
        #     print("%s with value %s" % (v.name, sess.run(v)))

        ## My Code
        prediction_result = np.empty((test_size, FLAGS.num_classes), dtype=np.float32)
        t_valid = np.empty((tnum, 4), dtype=np.float32)
        index = 0
        # for i in range(FLAGS.num_epoch):
        if (test_size < FLAGS.batch_size):
            prediction_result = sess.run(Ylogits,
                                         feed_dict = {X: test_data, is_training: False, keep_prob: 1})
        print (prediction_result)

        for s in range(tnum):
            if s == 29:
                print ("ERR")
            t_valid[s, :] = np.max(prediction_result[index:index + pernums_test[s], :], 0)
            index = index + pernums_test[s]
        # print (t_valid)
        # print (np.argmax(t_valid, 1))
        prediction_result = np.argmax(t_valid, 1)
        # print (type(a))
        return prediction_result

# test use
def read_random_csv():
    pernums_test = np.array([1])
    test_data = pd.read_csv('../random.csv', sep=',', header=None)
    test_data = np.array(test_data)
    predict(test_data, pernums_test)

# save to csv
def save_csv(name, data):
    with open('/media/mmnlab/mmndata/audio_emotion_result/speech_emotion_recognition_RNN_2/' + str(name) + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for rows in data:
            writer.writerow(rows)

# 統計各情緒數量
def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

# ===== start speech emotion recognition =====
def start_rec_emotion(audio_path, conn):
    # audio_path = "/media/mmnlab/mmndata/audio/wav_files/EMO_DB/" + sys.argv[1]  # audio file path
    mean1, std1, mean2, std2, mean3, std3 = audio_zscore(audio_path)
    test_data, pernums_test = read_audio(mean1, std1, mean2, std2, mean3, std3, audio_path)
    p_result = predict(test_data, pernums_test)
    print (p_result)  # 統計陣列裡最多的數值

    if (p_result == 0):
        emo_result = 'Angry'
        print(emo_result)
    elif (p_result == 1):
        emo_result = 'Sad'
        print(emo_result)
    elif (p_result == 2):
        emo_result = 'Happy'
        print(emo_result)
    elif (p_result == 3):
        emo_result = 'Neutral'
        print(emo_result)
    elif (p_result == 4):
        emo_result = 'Fear'
        print(emo_result)
    send_result_pi = threading.Thread(target=send_rec_result, args=(emo_result, conn))
    send_result_pi.start()
    send_result_web = threading.Thread(target=ws_send, args=(emo_result, ))
    send_result_web.start()

# ===== save audio file =====
def new_connect(conn, addr):
    global BASE_DIR, list_conn
    while True:
        data = conn.recv(1024)

        if (str(data).encode('utf-8') == "done" or str(data).encode('utf-8') == ''):
            return

        else:
            print(str(data).encode('utf-8'))
            cmd, filename, file_size = str(data).encode('utf-8').split('|')
            path = os.path.join(BASE_DIR, 'data', filename)
            file_size = int(file_size)
            has_sent = 0

            with open(path, 'wb') as fp:
                while has_sent != file_size:
                    data = conn.recv(1024)
                    fp.write(data)
                    has_sent += len(data)
                    print('\r' + '[保存進度]:%s%.02f%%' % ('>' * int((has_sent / file_size) * 50), float(has_sent / file_size) * 100), end='')
            print()
            print(' %s 保存成功！' % filename)
            print(path)
            start_rec = threading.Thread(target=start_rec_emotion, args=(path, conn))
            start_rec.start()

# ===== send recognition result to client(pi) =====
def send_rec_result(rec_result, conn):
    conn.sendall(bytes(rec_result).decode('utf-8'))

# ===== send web signal to client(pi) =====
def rs_web_cmd(task_cmd):
    global list_conn
    print(list_conn)
    for ip_index in range(len(list_conn)):
        list_conn[ip_index].sendall(bytes(task_cmd).decode('utf-8'))

# ===== websocket connect =====
def ws_connect():
    global ws, uri
    ws = websocket.WebSocket()
    ws.connect(uri)

# ===== websocket receive (receive web signal) =====
def ws_recv():
    global ws
    try:
        while True:
            print("**************************")
            message = ws.recv()
            text_json = json.loads(message)['message']
            print(text_json + "\n")
            if text_json == "Task1":
                send_task_cmd = threading.Thread(target=rs_web_cmd, args=("Task1",))
                send_task_cmd.start()
    except:
        ws.close()

# ===== websocket send (send emotion result to web) =====
def ws_send(em_result):
    global ws
    ws.send(json.dumps({'message': em_result}))

# ===== MAIN =====
if __name__ == '__main__':
    ws_connect()
    ws_recv_thread = threading.Thread(target=ws_recv)
    ws_recv_thread.start()
    IP_PORT = ('0.0.0.0', 12345)
    sk = socket.socket()
    sk.bind(IP_PORT)
    sk.listen(1)
    print('服務開啟成功!\n等待連線...')
    while True:
        conn, addr = sk.accept()
        print("{0}, {1} 已連接！".format(addr[0], addr[1]))
        # only append pi connect
        if (addr[0] != '127.0.0.1'):
            list_conn.append(conn)

        t = threading.Thread(target=new_connect, args=(conn, addr))
        t.start()


    # # time start
    # time_start = time.time()
    #
    # # time end
    # time_end = time.time()
    # print "cost time :", (time_end - time_start)


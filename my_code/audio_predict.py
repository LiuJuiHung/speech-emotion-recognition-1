#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2017

@author: hxj
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
from acrnn1 import *
import cPickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os

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

## Original Code for Valid
# tf.app.flags.DEFINE_string('testdata_path', './IEMOCAP.pkl', 'total dataset includes training set')

# tf.app.flags.DEFINE_string('validdata_path', 'inputs/valid.pkl', 'total dataset includes valid set')
tf.app.flags.DEFINE_string('checkpoint', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('model_name', 'model4.ckpt', 'model name')

FLAGS = tf.app.flags.FLAGS


def load_data(in_dir):
    f = open(in_dir, 'rb')

    ## My code
    test_data, pernums_test = cPickle.load(f)
    return test_data, pernums_test

    ## Original Code for Valid
    # train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = cPickle.load(f)
    # return train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def train():
    #####load data##########

    ## My code
    test_data, pernums_test = load_data(FLAGS.testdata_path)
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
        # load_model = tf.train.import_meta_graph('./checkpoint/model4.ckpt-1126.meta')
        # load_model.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './checkpoint30/model4.ckpt-3146')

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
        print (t_valid)
        print (np.argmax(t_valid, 1))


        ## Original Code for Valid
        # for i in range(FLAGS.num_epoch):
        #     if i % 5 == 0:
        #         # for valid data
        #         valid_iter = divmod((valid_size), FLAGS.batch_size)[0]
        #         y_pred_valid = np.empty((valid_size, FLAGS.num_classes), dtype=np.float32)
        #         y_valid = np.empty((vnum, 4), dtype=np.float32)
        #         index = 0
        #         cost_valid = 0
        #         if (valid_size < FLAGS.batch_size):
        #             y_pred_valid = sess.run([Ylogits],
        #                                           feed_dict={X: valid_data, is_training: False,
        #                                                      keep_prob: 1})
        #     for s in range(vnum):
        #         if s == 29:
        #             print("29")
        #         y_valid[s, :] = np.max(y_pred_valid[index:index + pernums_valid[s], :], 0)
        #         index = index + pernums_valid[s]
        #         print (y_pred_valid)

if __name__ == '__main__':
    train()

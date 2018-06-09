# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import jieba
import os
import numpy as np

from module.qa_preload import QAPreload as QA

import importlib
import sys
importlib.reload(sys)


cur_dir = os.getcwd()

MAX_LENTH = 16
OUT_SIZE1 = int(MAX_LENTH / 4)
OUT_SIZE2 = int(MAX_LENTH / 2)
CLASS_TYPE = 2
SAVERPATH = '{}/qa_data/pre_trained_models/CNN_{}/CNN_{}.ckpt'.format(cur_dir, MAX_LENTH, MAX_LENTH)


def sen_vector_gen(title_words):
    sen_vector = np.zeros(60, dtype=float)
    length = 0
    for word in title_words:
        try:
            sen_vector += QA.word_vectors[word]
            length += 1
        except:
            try:
                for w in jieba.lcut(word):
                    sen_vector += QA.word_vectors[w]
                    length += 1
            except:
                pass
    if length != 0:
        sen_vector = sen_vector / length
    return [sen_vector]


def get_vec_cosine(vec1, vec2):
    tmp = np.vdot(vec1, vec1) * np.vdot(vec2, vec2)
    if tmp == 0.0:
        return 0.0
    return np.vdot(vec1, vec2) / math.sqrt(tmp)


def s1_s2_simipics(s1_list, s2_list, max_lenth):
    k = 0
    simi = []
    while k < max_lenth:
        try:
            sen_k = sen_vector_gen(s1_list[k])
            j = 0
            while j < max_lenth:
                try:
                    sen_j = sen_vector_gen(s2_list[j])
                    simi_pic = get_vec_cosine(sen_k, sen_j)
                except:
                    simi_pic = 0.0
                simi.append(simi_pic)
                j += 1
        except:
            simi_pic = 0.0
            simi.append(simi_pic)
        k += 1
    while len(simi) < MAX_LENTH**2:
        simi.append(0.0)
    return simi


# ------------------CNN------------------ #


def weight_variable(shape, var_name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=var_name)
    return tf.Variable(initial)


def bias_variable(shape, var_name):
    initial = tf.constant(0.1, shape=shape, name=var_name)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class CNNSentenceSimilarity(object):
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [MAX_LENTH ** 2], 'x_input')
    x_image = tf.reshape(xs, [-1, MAX_LENTH, MAX_LENTH, 1])

    W_conv1 = weight_variable([5, 5, 1, OUT_SIZE1], 'w1')
    b_conv1 = bias_variable([OUT_SIZE1], 'b1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, OUT_SIZE1, OUT_SIZE2], 'w2')
    b_conv2 = bias_variable([OUT_SIZE2], 'b2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([OUT_SIZE1 * OUT_SIZE1 * OUT_SIZE2, MAX_LENTH], 'wf1')
    b_fc1 = bias_variable([MAX_LENTH], 'bf1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, OUT_SIZE1 * OUT_SIZE1 * OUT_SIZE2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([MAX_LENTH, CLASS_TYPE], 'wf2')
    b_fc2 = bias_variable([CLASS_TYPE], 'bf2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    @classmethod
    def calculate_similarity(cls, s1, s2):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, SAVERPATH)
            prediction_result = sess.run(
                CNNSentenceSimilarity.prediction,
                feed_dict={
                    CNNSentenceSimilarity.xs: s1_s2_simipics(jieba.lcut(s1), jieba.lcut(s2), MAX_LENTH),
                    CNNSentenceSimilarity.keep_prob: 1.0
                }
            )

        return prediction_result.tolist()[0][-1]


if __name__ == '__main__':
    while True:
        s1 = input("请输入句子1：")
        s2 = input("请输入句子2：")

        print(CNNSentenceSimilarity.calculate_similarity(s1, s2))

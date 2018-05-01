# -*- coding: utf-8 -*-

import jieba
import tensorflow as tf
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split
from gensim.models import keyedvectors

import importlib, sys
importlib.reload(sys)

# ------------------预加载------------------ #

cur_dir = os.getcwd()
print(cur_dir)

# 核心词表
all_keywords = set(list(open(cur_dir + '/qa_data/basic_data/keywords.txt', 'r').read().strip().split('\n')))
for keyword in all_keywords:
    jieba.add_word(keyword, freq=None, tag=None)

# 搜狗细胞词库
field_words = set(list(open(cur_dir + '/qa_data/basic_data/fieldwords.txt', 'r').read().strip().split('\n')))
for field_word in field_words:
    jieba.add_word(field_word, freq=None, tag=None)

# 加载预先训练好的词向量，基于10G中文维基训练的60维词向量
word_vectors = keyedvectors.KeyedVectors.load(cur_dir + '/qa_data/word_vectors/word60.model')

# CNN相关参数
MAX_LENTH = 16  # 训练时保留的词的最大数量, 必须为4的倍数
OUT_SIZE1 = int(MAX_LENTH / 4)
OUT_SIZE2 = int(MAX_LENTH / 2)
CLASS_TYPE = 2
PATH = '{}/qa_data/pre_trained_models/CNN_{}'.format(cur_dir, MAX_LENTH)
if os.path.exists(PATH) is False:
    os.makedirs(PATH)


# ------------------基础函数------------------ #

def sen_vector_gen(title_words):
    sen_vector = np.zeros(60, dtype=float)
    length = 0
    for word in title_words:
        try:
            sen_vector += word_vectors[word]
            length += 1
        except:
            try:
                for w in jieba.lcut(word):
                    sen_vector += word_vectors[w]
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


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


if __name__ == '__main__':

    with tf.name_scope('inputs'):
        keep_prob = tf.placeholder(tf.float32)
        xs = tf.placeholder(tf.float32, [None, MAX_LENTH ** 2], 'x_input')
        ys = tf.placeholder(tf.float32, [None, CLASS_TYPE], 'y_input')
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

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('CNN_logs/', sess.graph)
        sess.run(init)

        s = np.zeros(MAX_LENTH ** 2 + 2, float)
        for line in open(cur_dir + '/qa_data/basic_data/cnn_train_data.txt', 'r'):
            line_seg = line.strip().split('\t')
            w1 = line_seg[1]
            w2 = line_seg[3]
            label1 = line_seg[4]
            label2 = line_seg[5]
            line_simi = s1_s2_simipics(w1, w2, MAX_LENTH) + [float(label1)] + [float(label2)]
            s = np.vstack((s, line_simi))
        S = s[1:, :]
        np.random.shuffle(S)
        X_train, X_test, Y_train, Y_test = train_test_split(S[:, :-2], S[:, -2:], test_size=0.1)

        for i in range(1000):  # 训练
            # batch_xs, batch_xy = X_train[5 * i:5 * i + 400, :], Y_train[5 * i:5 * i + 400, :]
            batch_xs, batch_xy = X_train, Y_train
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_xy, keep_prob: 0.8})
            if i % 50 == 0:
                result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_xy, keep_prob: 1})
                writer.add_summary(result, i)
                print(compute_accuracy(X_test, Y_test))

        save_path = saver.save(sess, '{}/CNN_{}.ckpt'.format(PATH, MAX_LENTH))
        print("模型保存到路径:" + save_path)

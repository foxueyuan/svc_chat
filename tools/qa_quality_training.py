# -*- coding: utf-8 -*-

import math
import jieba
import os
import jieba.posseg as pseg
import numpy as np
import re
import pickle
from snownlp import SnowNLP
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

import importlib, sys
importlib.reload(sys)


cur_dir = os.getcwd()

# 停用词表
stopwords = set(list(open(cur_dir + '/qa_data/basic_data/stopwords.txt', 'r').read().strip().split('\n')))

# 核心词表
all_keywords = set(list(open(cur_dir + '/qa_data/basic_data/keywords.txt', 'r').read().strip().split('\n')))
for keyword in all_keywords:
    jieba.add_word(keyword, freq=None, tag=None)

# 搜狗细胞词库
field_words = set(list(open(cur_dir + '/qa_data/basic_data/fieldwords.txt', 'r').read().strip().split('\n')))
for field_word in field_words:
    jieba.add_word(field_word, freq=None, tag=None)


class QAQuality:
    def __init__(self, q, a):
        self.answer = q
        self.question = a
        self.alenth = len(self.answer)
        self.qlenth = len(self.question)
        self.q_pos_list = pseg.lcut(self.question)
        self.a_pos_list = pseg.lcut(self.answer)
        self.inter_list = [val for val in self.q_pos_list if val in self.a_pos_list]
        self.union_list = list(set(self.q_pos_list + self.a_pos_list))
        self.q_noun = [qnoun.word for qnoun in self.q_pos_list if 'n' in str(qnoun)]
        self.a_noun = [anoun.word for anoun in self.a_pos_list if 'n' in str(anoun)]
        self.qa_noun = self.q_noun + self.a_noun
        self.q_verb = [qverb.word for qverb in self.q_pos_list if 'v' in str(qverb)]
        self.a_verb = [averb.word for averb in self.a_pos_list if 'v' in str(averb)]
        self.qa_verb = self.q_verb + self.a_verb
        self.a_stop = [stopword.word for stopword in self.a_pos_list if stopword.word in stopwords]
        self.q_nonpunc = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', self.question))
        self.a_nonpunc = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', self.answer))
        self.qa_nonpunc = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', self.question + self.answer))
        self.a_singleword = ['@&@' for a in self.a_pos_list if len(a.word) == 1].count('@&@')
        self.n_sentence = (len(self.answer.strip().split('。')) - 1 if '。' in self.answer else 1)

        self.lenth_ratio = float(len(self.answer)) / float(len(self.question))
        self.resemblance = float(len(self.inter_list)) / float(len(self.union_list))
        self.contain = float(len(self.inter_list)) / float(len(self.a_pos_list))
        self.overlap = float(len(self.a_pos_list)) / float(len(self.q_pos_list))
        self.cosine = float(len(self.inter_list)) / math.sqrt((len(self.q_pos_list)*len(self.a_pos_list)))
        self.qa_noun_per = float(len(self.qa_noun)) / float(len(self.q_pos_list) + len(self.a_pos_list))
        self.a_noun_per = float(len(self.a_noun)) / float(len(self.a_pos_list))
        self.qa_verb_per = float(len(self.qa_verb)) / float(len(self.q_pos_list) + len(self.a_pos_list))
        self.a_verb_per = float(len(self.a_verb)) / float(len(self.a_pos_list))
        self.a_stop_per = float(len(self.a_stop)) / float(len(self.a_pos_list))
        self.a_punc_dens = 1.0 - float(len(self.a_nonpunc)) / float(len(self.answer))
        self.qa_punc_dens = 1.0 - float(len(self.qa_nonpunc)) / float(len(self.question + self.answer))
        self.a_sentiment = SnowNLP(self.answer).sentiments
        self.a_singleword_per = float(self.a_singleword) / float(len(self.a_pos_list))
        self.q_punc_dens = 1.0 - float(len(self.q_nonpunc)) / float(len(self.question))
        self.fog_score = 0.4 * (float(len(self.a_pos_list)) / float(self.n_sentence) + 100 * (
            float(len(self.a_pos_list) - self.a_singleword)) / (float(len(self.a_pos_list))))
        self.flesch_score = 206.835 * (1.015 * float(len(self.a_pos_list)) / float(self.n_sentence)) * (
            84.6 * float(sum(len(a.word) for a in self.a_pos_list)) / len(self.a_pos_list))
        self.flesch_kincaid_score = 0.39 * (float(len(self.a_pos_list)) / float(self.n_sentence)) + 11.8 * (
            float(sum(len(a.word) for a in self.a_pos_list)) / len(self.a_pos_list)) - 15.59

    def a_wentroy(self):
        num_dict = {}
        for item in [l.word for l in self.a_pos_list]:
            num_dict[item] = num_dict.get(item, 0.0) + 1.0
        a_dic = {k: v / float(len(self.a_pos_list)) for k, v in num_dict.items()}
        entroy = 0.0
        for key in a_dic:
            entroy += a_dic[key]*(math.log(float(len(self.a_pos_list)), 10)-math.log(a_dic[key], 10))
        return entroy / float(len(self.a_pos_list))

    def a_centroy(self):
        num_dict = {}
        for item in list(self.answer):
            num_dict[item] = num_dict.get(item, 0.0) + 1.0
        a_dic = {k: v / float(self.alenth) for k, v in num_dict.items()}
        entroy = 0.0
        for key in a_dic:
            entroy += a_dic[key]*(math.log(float(self.alenth), 10)-math.log(a_dic[key], 10))
        return entroy / float(self.alenth)

    def a_singles(self):
        maxflag = 0
        countflag = 0
        for apw in [l.word for l in self.a_pos_list]:
            if len(apw) == 1:
                countflag += 1
                maxflag = countflag
            else:
                countflag = 0
        return maxflag


if __name__ == '__main__':

    train_path = cur_dir + '/qa_data/basic_data/qar_test.txt'

    # 生成特征矩阵
    X = np.zeros(22, float)

    for line in open(train_path, mode='r', encoding='utf-8'):
        question = line.strip().split('\t')[0]
        answer = line.strip().split('\t')[1]
        label = line.strip().split('\t')[2]

        ac = QAQuality(question, answer)
        features = [float(ac.a_singles()), ac.lenth_ratio, ac.resemblance, ac.contain, ac.overlap, ac.cosine,
                    ac.qa_noun_per, ac.a_noun_per, ac.qa_verb_per, ac.a_verb_per, ac.a_stop_per, ac.a_punc_dens,
                    ac.qa_punc_dens, ac.a_sentiment, ac.a_singleword_per, ac.a_wentroy(), ac.a_centroy(),
                    ac.q_punc_dens, ac.fog_score, ac.flesch_score, ac.flesch_kincaid_score, float(label)]

        X = np.vstack((X, features))

    S = X[1:, :]

    # 训练
    i = 0
    trainning_accuracy = 0
    test_accuracy = 0
    trainning_max = 0
    test_max = 0
    while i < 100:  # 训练10轮，观测每一轮十折交叉验证的准确率

        # 拆分训练集与测试集
        np.random.shuffle(S)  # 随机洗牌
        train_X, test_X, train_Y, test_Y = train_test_split(S[:, :-1], S[:, -1], test_size=0.1)  # 十折分割训练集
        aTrain_X = train_X  # .as_matrix()
        aTrain_Y = train_Y  # .as_matrix()
        aTest_X = test_X  # .as_matrix()
        aTest_Y = test_Y  # .as_matrix()

        # 使用xgboost模型
        # 模型参数
        params = {
            'booster': 'gbtree',
            # 'objective': 'multi:softmax', #多分类的问题
            # //无效'num_class':2, # 类别数，与 multisoftmax 并用
            'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 8,  # 构建树的深度，越大越容易过拟合
            # //无效'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            # //无效'eta': 0.007, # 如同学习率
            'seed': 1000,
            'nthread': 7,  # cpu 线程数
            # 'eval_metric': 'auc'
        }

        # 训练模型
        model = XGBClassifier()  # 构建模型
        model.get_params()       #获取参数
        model.set_params(**params)  # 设置参数
        # 开始训练
        model.fit(aTrain_X, aTrain_Y, eval_metric='auc')

        # 保存模型
        score0 = 0  # model.score(aTrain_X, aTrain_Y)
        score1 = model.score(aTest_X, aTest_Y)
        if score1 > 0.745:
            pickle.dump(
                model,
                open('{}/qa_data/pre_trained_models/xgboost_qaquality_21_60dz_s{}.pkl'.format(cur_dir,
                                                                                              round(score1, 3)),
                     'wb')
            )
            print('====> yes found good xgboost model')
        # print(i+1, score)  # 打印每轮训练的准确率
        # 打印准确率 和 召回率
        print('第 %d 轮测试集计算准确率为：%f' % (i, score1))
        trainning_accuracy += score0
        test_accuracy += score1
        trainning_max = max(score0, trainning_max)
        test_max = max(score1, test_max)
        i += 1

    trainning_accuracy /= i
    test_accuracy /= i
    print("平均训练集准确率: %f. 最大训练集准确率 %f" % (trainning_accuracy, trainning_max))
    print("平均测试集准确率: %f. 最大测试集准确率 %f" % (test_accuracy, test_max))

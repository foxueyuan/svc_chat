# -*- coding: utf-8 -*-

import jieba.posseg as pseg
from snownlp import SnowNLP
import re
import math

from module.qa_preload import QAPreload as QA

import importlib, sys
importlib.reload(sys)


class QAQuality:
    def __init__(self, Q, A):
        self.answer = A
        self.question = Q
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
        self.a_stop = [stopword.word for stopword in self.a_pos_list if stopword.word in QA.stopwords]
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

    def answer_judge_pro(self):
        features = [float(self.a_singles()), self.lenth_ratio, self.resemblance, self.contain, self.overlap,
                    self.cosine, self.qa_noun_per, self.a_noun_per, self.qa_verb_per, self.a_verb_per, self.a_stop_per,
                    self.a_punc_dens, self.qa_punc_dens, self.a_sentiment, self.a_singleword_per, self.a_wentroy(),
                    self.a_centroy(), self.q_punc_dens, self.fog_score, self.flesch_score, self.flesch_kincaid_score]
        return QA.xgb.predict_proba(features)[:, -1].tolist()


if __name__ == '__main__':
    while True:
        q_input = input('请输入问题：')
        a_input = input('请输入回答：')
        print(float(QAQuality(q_input, a_input).answer_judge_pro()[0]))

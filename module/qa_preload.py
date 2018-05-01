# -*- coding: utf-8 -*-

import jieba
import os
import pickle
from gensim.models import keyedvectors
from pyltp import NamedEntityRecognizer, Postagger, Segmentor, Parser  # 用于抽取句子NER和主谓宾主干成分

import importlib, sys
importlib.reload(sys)


class QAPreload(object):
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

    # 加载预先训练好的词向量，基于10G中文维基训练的60维词向量
    word_vectors = keyedvectors.KeyedVectors.load(cur_dir + '/qa_data/word_vectors/word60.model')

    # 预测问答对属于"好"的概率的问答质量评价子模块
    xgb = pickle.load(open(cur_dir + '/qa_data/pre_trained_models/xgboost_qaquality_21_60dz_s0.745.pkl', 'rb'))

    LTP_DATA_DIR = cur_dir + '/qa_data/ltp_data_v3.4.0'
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径
    pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")  # 词性模型路径
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")  # 分词模型路径
    par_model_path = os.path.join(LTP_DATA_DIR, "parser.model")  # parser模型路径
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型
    postagger = Postagger()
    postagger.load(pos_model_path)
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    parser = Parser()
    parser.load(par_model_path)

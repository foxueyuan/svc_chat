# -*- coding: utf-8 -*-

import time
import jieba
import json

from module.qa_preload import QAPreload as QA
from module.qa_es import ElasticSearchClient, query_generation
from module.qa_quality import QAQuality
from module.qa_cnn_sentence_similarity import CNNSentenceSimilarity

import importlib, sys
importlib.reload(sys)


THRESHOLD = 1.0
SIMI = 1.0
QUA = 1.0


def ner(words):  # 命名实体识别
    postags = QA.postagger.postag(words)
    netags = QA.recognizer.recognize(words, postags)
    return [words[list(netags).index(netag)] for netag in list(netags) if netag != "O"]


# 以下三个函数实现了主谓宾三元组抽取
def build_parse_child_dict(words, arcs):  # 依存句法分析函数
    """
    为句子中的每个词语维护一个保存句法依存儿子节点的字典
    Args:
        words: 分词列表
        arcs: 句法依存列表
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                if arcs[arc_index].relation in child_dict:
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)
        child_dict_list.append(child_dict)
    return child_dict_list


def complete_e(words, postags, child_dict_list, word_index):  # 完善识别的部分实体
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])

    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix


def fact_triple_extract(sentence):  # 对于给定的句子进行事实三元组抽取
    words = QA.segmentor.segment(sentence)
    postags = QA.postagger.postag(words)
    arcs = QA.parser.parse(words, postags)

    child_dict_list = build_parse_child_dict(words, arcs)
    for index in range(len(postags)):
        # 抽取以谓词为中心的事实三元组
        if postags[index] == 'v':
            child_dict = child_dict_list[index]
            # 主谓宾
            if 'SBV' in child_dict and 'VOB' in child_dict:
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                r = words[index]
                e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                return [e1, r, e2]
    return []


def word_weight_exchange(original_list, original_text):  # 修改列表中各个词的词频权重

    weighted_list = original_list
    key_tempt_list = []
    for word in original_list:  # 提高重点词的权重
        if word in QA.all_keywords:
            key_tempt_list.append(word)
    weighted_list += key_tempt_list
    synon_tempt_list = []
    for synon in original_list:  # 添加部分近义词
        if synon in QA.word_vectors:
            synon_tempt_list.append(QA.word_vectors.most_similar(synon)[0][0])
    weighted_list += synon_tempt_list
    weighted_list += ner(original_list)  # 提高NER权重
    weighted_list += fact_triple_extract(original_text)  # 提高主干部分权重
    final_list = [fw for fw in weighted_list if fw not in QA.stopwords]

    return final_list


def question_answer(question):  # 用户输入一句新问题

    return_dic = {}  # 本函数最终返回的精选问答对

    # 经加权后的分词、去停用词后的用户新问题主要内容
    text_main_content = word_weight_exchange(jieba.lcut(question), question)

    # 封装成es query需要的样式，match_phrase可保证分词后的词组不被再拆散成字
    es = ElasticSearchClient.get_es_servers()
    input_string = [{"match_phrase": {"question": phrase}} for phrase in text_main_content]
    _query_name_contains = query_generation(input_string)  # 生成ES查询
    _searched = es.search(index='my-index', doc_type='test-type', body=_query_name_contains)

    # 无查询结果，直接返回空，稍后调用闲聊接口
    if len(_searched['hits']['hits']) == 0:
        return json.dumps(return_dic)

    _selected = {}  # 存放精选后的候选问题-答案的集合
    for l1 in _searched['hits']['hits']:  # 问题精选

        # 如果新问题与候选问题完全一致，则直接返回
        if l1['_source']['question'] == question:
            return_dic[l1['_source']['question']] = l1['_source']['answer']
            return json.dumps(return_dic)

        # 生成与新问题、候选问题相关的基础运算变量
        qa_pair = (l1['_source']['question'], l1['_source']['answer'])
        # 经加权的分词、去停用词后的候选问题主要内容
        question_main_content = word_weight_exchange(jieba.lcut(l1['_source']['question']), l1['_source']['question'])

        # 新问题与候选问题句子相似度similarity
        similarity = CNNSentenceSimilarity.calculate_similarity(text_main_content, question_main_content)

        try:  # 答案排序模块对新问题和候选答案的组合给出的分数
            qa_quality_score = float(QAQuality(question, l1['_source']['answer']).answer_judge_pro()[0])
        except:
            # 有时由于答案过短，答案排序模块无法完成计算，则直接置0
            qa_quality_score = 0.0

        # 最终精选得分
        _selected[qa_pair] = SIMI * similarity + QUA * qa_quality_score

    # 精选top1
    selected = sorted(_selected.items(), key=lambda item: item[1], reverse=True)[0]
    if selected[1] > THRESHOLD:
        return_dic[selected[0][0]] = selected[0][1]
        return json.dumps(return_dic)

    return json.dumps(return_dic)


if __name__ == '__main__':
    t = 0
    k = 0
    while True:

        s_input = input('请输入：')
        start = time.time()
        result = json.loads(question_answer(s_input))
        for key in result:
            print(key, result[key])
        end = time.time()
        print("=" * 40)
        print("耗时：" + str(round((end - start), 3)) + "s")

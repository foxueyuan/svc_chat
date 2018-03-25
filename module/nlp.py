# -*- coding: utf-8 -*-

import jieba.posseg as pseg
from gensim import corpora, models, similarities
from simhash import Simhash, SimhashIndex


async def _tokenization(conf, text):
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in conf.DEMO_STOP_FLAG:
            result.append(word)
    return result


async def tokenization(conf, items):
    segs = []
    for item in items:
        words = pseg.cut(item)
        for word, flag in words:
            if flag not in conf.DEMO_STOP_FLAG and word not in conf.DEMO_STOPWORDS:
                segs.append(word)
    return segs


async def gen_corpus_vectors(conf):
    corpus = []
    for items in conf.DEMO_QUESTION:
        corpus.append(await tokenization(conf, items))

    dictionary = corpora.Dictionary(corpus)
    corpus_vectors = [dictionary.doc2bow(text) for text in corpus]

    return dictionary, corpus_vectors


async def simi_cal_tfidf(conf, corpus_vectors, dictionary, text):
    tfidf = models.TfidfModel(corpus_vectors)
    tfidf_vectors = tfidf[corpus_vectors]
    query = await tokenization(conf, text)
    query_bow = dictionary.doc2bow(query)
    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_bow]

    if list(sims) == [0.0] * len(sims):
        return []

    return list(enumerate(sims))


async def simi_cal_lsi(conf, corpus_vectors, dictionary, text):
    tfidf = models.TfidfModel(corpus_vectors)
    tfidf_vectors = tfidf[corpus_vectors]
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=300)
    lsi.print_topics(300)
    lsi_vector = lsi[tfidf_vectors]
    query = await tokenization(conf, text)
    query_bow = dictionary.doc2bow(query)
    query_lsi = lsi[query_bow]
    index = similarities.MatrixSimilarity(lsi_vector, num_features=100)
    sims = index[query_lsi]

    if list(sims) == [0.0] * len(sims):
        return []

    return list(enumerate(sims))


async def gen_simhash_index(conf):
    m = 0
    n = 0
    objs = []
    simhash_answer_index = {}
    for items in conf.DEMO_QUESTION:
        for item in items:
            objs.append((n, Simhash(await _tokenization(conf, item))))
            simhash_answer_index[n] = m
            n += 1
        m += 1

    simhash_index = SimhashIndex(objs, k=6)
    return simhash_index, simhash_answer_index


async def simi_cal_simhash(conf, simhash_index, text):
    s = Simhash(await _tokenization(conf, text))
    rst = simhash_index.get_near_dups(s)
    if not rst:
        return None
    else:
        return int(rst[0])

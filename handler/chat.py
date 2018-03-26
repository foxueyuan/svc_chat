# -*- coding: utf-8 -*-

import aiohttp
from sanic import response

from module.nlp import simi_cal_tfidf
from module.nlp import simi_cal_lsi
from module.nlp import simi_cal_simhash


async def chat(request):
    conf = request.app.config
    data = request.json

    result = {'errcode': 0, 'errmsg': 'ok'}

    if 'speech' in data or 'speech_url' in data:
        text_rst = await  request_asr(conf.SVC_ASR_URL, data.get('speech'), data.get('speech_url'))
        if text_rst['errcode'] != 0:
            return response.json({'errcode': 0,
                                  'errmsg': 'ok',
                                  'data': {'msgtype': 'text',
                                           'content': '对不起，我没有听清楚您说什么，请再说一遍。'}})
        else:
            data['text'] = text_rst['content']

    simhash_index = request.app.simhash_index
    simhash_answer_index = request.app.simhash_answer_index

    rst = await simi_cal_simhash(conf, simhash_index, data['text'])
    if rst:
        demo_answer_index = simhash_answer_index[rst]
        result['data'] = {'msgtype': 'text',
                          'content': conf.DEMO_ANSWER[demo_answer_index]}
        return response.json(result)

    corpus_vectors = request.app.corpus_vectors
    dictionary = request.app.dictionary

    # demo_simlarity_tfidf = await simi_cal_tfidf()
    demo_simlarity_lsi = await simi_cal_lsi(conf, corpus_vectors, dictionary, data['text'])

    if not demo_simlarity_lsi:
        result['data'] = {'msgtype': 'text',
                          'content': '对不起，我不明白你在说什么，但我很快就会明白。'}
        return response.json(result)

    demo_answer_index = sorted([(abs(v - 1), k) for k, v in demo_simlarity_lsi])[0][1]

    result['data'] = {'msgtype': 'text',
                      'content': conf.DEMO_ANSWER[demo_answer_index]}

    return response.json(result)


async def chat_with_asr_cb(request):
    data = request.json

    result = {'errcode': 0, 'errmsg': 'ok'}

    return response.json(result)


async def request_asr(url, speech=None, speech_url=None):
    if speech:
        payload = {'speech': speech}
    else:
        payload = {'url': speech_url}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()


async def request_unit(url, scene_id, text):
    payload = {'scene_id': scene_id, 'text': text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

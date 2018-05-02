# -*- coding: utf-8 -*-

import aiohttp
import json
from sanic import response

from module.qa_main import question_answer


async def qa_test(request):
    data = request.json

    result = question_answer(data['text'])

    return response.json(result)


async def chat(request):
    conf = request.app.config
    data = request.json

    result = {'errcode': 0, 'errmsg': 'ok'}

    if 'speech' in data or 'speech_url' in data:
        text_rst = await request_asr(conf.SVC_ASR_URL, data.get('speech'), data.get('len'), data.get('speech_url'))
        if text_rst['errcode'] != 0:
            return response.json({'errcode': 0,
                                  'errmsg': 'ok',
                                  'data': {'msgtype': 'text',
                                           'content': '对不起，我没有听清楚您说什么，请再说一遍。'}})
        else:
            data['text'] = text_rst['content']

    # 文本审核，违禁处理
    spam_resp = await request_spam(conf.SVC_SPAM_URL, data['text'])
    if spam_resp.get('spam', 0):
        result['data'] = {'msgtype': 'text',
                          'text': data['text'],
                          'content': '我们都是文明人，怎么可以说这样的话。'}
        return response.json(result)

    # 语义解析，初步识别意图
    wordcom_resp = await request_wordcom(conf.SVC_WORDCOM_URL, data['text'])
    if wordcom_resp.get('intent') == 1:
        for con_token in wordcom_resp.get('com_tokens', []):
            if con_token['com_type'] == 7:
                # 处理天气问答，涉及到具体地理，后续多轮对话实现
                weather_answer = await request_small_talk(conf.SVC_SMALL_TALK_URL, data['text'])
                result['data'] = {'msgtype': 'text',
                                  'text': data['text'],
                                  'content': weather_answer}

        if 'data' not in result:
            result['data'] = {'msgtype': 'text',
                              'text': data['text'],
                              'content': '需要说明查询地区，您可以这么问："长沙今天天气怎样？"'}
        return response.json(result)

    if len(data['text']) < 6:
        # 问句太短直接进入闲聊模式
        small_talk_answer = await request_textchat(conf.SVC_TEXTCHAT_URL, data['text'])
        result['data'] = {'msgtype': 'text',
                          'text': data['text'],
                          'content': small_talk_answer}
        return response.json(result)

    # 从问答库和知识库中匹配
    q_a_hint = question_answer(data['text'])

    if q_a_hint:
        result['data'] = {'msgtype': 'text',
                          'text': data['text'],
                          'content': q_a_hint['answer']}
    else:
        # 匹配不到进入闲聊模式
        small_talk_answer = await request_textchat(conf.SVC_TEXTCHAT_URL, data['text'])
        result['data'] = {'msgtype': 'text',
                          'text': data['text'],
                          'content': small_talk_answer}

    return response.json(result)


async def chat_with_asr_cb(request):
    result = {'errcode': 0, 'errmsg': 'ok'}
    return response.json(result)


async def request_asr(url, speech=None, len=None, speech_url=None):
    if speech:
        payload = {'speech': speech, 'len': len}
    else:
        payload = {'url': speech_url}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()


async def request_spam(url, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'text': text}) as resp:
            return await resp.json()


async def request_unit(url, scene_id, text):
    payload = {'scene_id': scene_id, 'text': text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()


async def request_small_talk(url, ask):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params={'ask': ask}) as resp:
            return await resp.text()


async def request_wordcom(url, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'text': text}) as resp:
            return await resp.json()


async def request_textchat(url, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'text': text}) as resp:
            resp_json = await resp.json()
            return resp_json['answer']

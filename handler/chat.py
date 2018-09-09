# -*- coding: utf-8 -*-

import aiohttp
import json
import time
import jieba
from sanic import response

from log_task import log_task

# from module.qa_main import question_answer


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
                                           'content': {'text': '对不起，我没有听清楚您说什么，请再说一遍。'}}})
        else:
            data['text'] = text_rst['content']

    # 文本审核，违禁处理
    spam_resp = await request_spam(conf.SVC_SPAM_URL, data['text'])
    if spam_resp.get('spam', 0):
        result['data'] = {'msgtype': 'text',
                          'text': data['text'],
                          'content': {'text': '我们都是文明人，怎么可以说这样的话。'}}
        return response.json(result)

    # 语义解析，初步识别意图
    wordcom_resp = await request_wordcom(conf.SVC_WORDCOM_URL, data['text'])
    if wordcom_resp.get('intent') == 1:
        for con_token in wordcom_resp.get('com_tokens', []):
            if con_token['com_type'] == 7 and (con_token['com_word'] != '深圳' or '深圳' in data['text']):
                # 处理天气问答，涉及到具体地理，后续多轮对话实现
                weather_answer = await request_small_talk(conf.SVC_SMALL_TALK_URL, data['text'])
                result['data'] = {'msgtype': 'text',
                                  'text': data['text'],
                                  'content': {'text': weather_answer}}

        if 'data' not in result:
            result['data'] = {'msgtype': 'text',
                              'text': data['text'],
                              'content': {'text': '需要说明查询地区，您可以这么问："长沙今天天气怎样"'}}
        return response.json(result)

    kg_rst = await request_unit(conf.SVC_UNIT_CHAT_URL, data['text'])
    if kg_rst['errcode'] == 0:
        for action in kg_rst['result']['response']['action_list']:
            if action['type'] == 'satisfy':
                if action['action_id'] == 'faq_instruction_satisfy':
                    pass
                if action['say'].startswith('#instruction#'):
                    content = json.loads(action['say'].strip('#instruction#'))
                    if content['action'] == 'time':
                        result['data'] = {'msgtype': 'instruction',
                                          'text': data['text'],
                                          'content': time.strftime('%Y年%d月%m日 %H点%M分%S秒', time.localtime())}
                    else:
                        result['data'] = {'msgtype': 'instruction',
                                          'text': data['text'],
                                          'content': content}
                else:
                    result['data'] = {'msgtype': 'text',
                                      'text': data['text'],
                                      'content': {'text': action['say']}}
                if action['action_id'] == 'faq_qa_satisfy':
                    log_task.send(data['text'], action['say'], 'qa')
                elif action['action_id'] == 'faq_kg_satisfy':
                    log_task.send(data['text'], action['say'], 'kg')
                elif action['action_id'] == 'faq_instruction_satisfy':
                    log_task.send(data['text'], action['say'], 'instruction')
                return response.json(result)

    # 匹配不到进入闲聊模式
    small_talk_answer = await request_small_talk(conf.SVC_SMALL_TALK_URL, data['text'])
    result['data'] = {'msgtype': 'text',
                      'text': data['text'],
                      'content': {'text': small_talk_answer}}

    log_task.send(data['text'], small_talk_answer, 'smalltalk')

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


async def request_unit(url, text):
    payload = {'text': text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()


async def request_wordcom(url, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'text': text}) as resp:
            return await resp.json()


async def request_small_talk(url, text):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'text': text}) as resp:
            resp_json = await resp.json()
            return resp_json['answer']

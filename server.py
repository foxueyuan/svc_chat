# -*- coding: utf-8 -*-

import asyncio
import aioredis
import uvloop
import jieba
from sanic import Sanic

import config

from handler.chat import chat


app = Sanic(__name__)
app.config.from_object(config)


app.add_route(chat, '/chat', methods=['POST'])
app.add_route(chat, '/chat_with_asr_cb', methods=['POST'])


@app.listener('before_server_start')
async def before_server_start(app, loop):
    conf = app.config
    app.rdb = await aioredis.create_redis_pool(
        (conf.REDIS_HOST, conf.REDIS_PORT),
        db=conf.REDIS_DB,
        encoding='utf8',
        loop=loop
    )

    async for val in app.rdb.isscan('keywords', match='*'):
        jieba.add_word(val)

@app.listener('after_server_stop')
async def after_server_stop(app, loop):
    app.rdb.close()
    await app.rdb.wait_closed()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    app.run(host=app.config.HOST, port=app.config.PORT)

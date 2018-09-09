import dramatiq
import time
from elasticsearch import Elasticsearch
from dramatiq.brokers.redis import RedisBroker
# from dramatiq.results.backends import RedisBackend
# from dramatiq.results import Results

import config as conf

# result_backend = RedisBackend(url="redis://127.0.0.1:16379/14")
redis_broker = RedisBroker(url="redis://127.0.0.1:16379/15")
# redis_broker.add_middleware(Results(backend=result_backend))
dramatiq.set_broker(redis_broker)

es = Elasticsearch(hosts=conf.ES_HOST)


@dramatiq.actor
def log_task(question, answer):
    body = {
        "question": question,
        "answer": answer,
        "createdAt": int(time.time())
    }

    es.update(index='foai-log-index', doc_type='chat', body=body)

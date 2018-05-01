# -*- coding: utf-8 -*-

import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


cur_dir = os.getcwd()

ESMAX = 10


def query_generation(inputs):
    _query = {
        'query': {
            'bool': {
                'should': inputs
            }
        },
        'size': ESMAX
    }

    '''# multi-match多级查询例子
    {
        "multi_match": {
            "query": {
                "bool": {
                    "should": inputs
                }
            }
        }
    }
    '''

    '''# bool布尔查询例子
    {
        "bool": {
            "must":     { "match": { "title": "how to make millions" }},
            "must_not": { "match": { "tag":   "spam" }},
            "should": [
                { "match": { "tag": "starred" }},
                { "range": { "date": { "gte": "2014-01-01" }}}
            ]
        }
    }
    '''

    return _query


class ElasticSearchClient(object):
    @staticmethod
    def get_es_servers():
        es_servers = [{
            "host": "localhost",
            "port": "9200"
        }]
        es_client = Elasticsearch(hosts=es_servers)
        return es_client


class LoadElasticSearch(object):  # 在ES中加载、批量插入数据
    def __init__(self):
        self.index = 'fo-qa'
        self.doc_type = 'test-type'
        self.es_client = ElasticSearchClient.get_es_servers()
        self.set_mapping()

    def set_mapping(self):
        mapping = {
            self.doc_type: {
                    'topic': {
                        'type': 'string'
                    },
                    'title': {
                        'type': 'string'
                    },
                    'question': {
                        'type': 'string'
                    },
                    'answer': {
                        'type': 'string'
                    }
                }
            }

        if not self.es_client.indices.exists(index=self.index):
            self.es_client.indices.create(index=self.index, body=mapping, ignore=400)
            self.es_client.indices.put_mapping(index=self.index, doc_type=self.doc_type, body=mapping)

    def add_data(self, row_obj):
        """
        单条插入ES
        """
        _id = row_obj.get('_id', 1)
        row_obj.pop('_id')
        self.es_client.index(index=self.index, doc_type=self.doc_type, body=row_obj, id=_id)

    def add_data_bulk(self, row_obj_list):
        """
        批量插入ES
        """
        load_data = []
        i = 1
        bulk_num = 10000
        for row_obj in row_obj_list:
            action = {
                '_index': self.index,
                '_type': self.doc_type,
                '_source': {
                    'topic': row_obj.get('topic', None),
                    'question': row_obj.get('question', None),
                    'answer': row_obj.get('answer', None),
                }
            }
            load_data.append(action)
            i += 1
            # 批量处理
            if len(load_data) == bulk_num:
                print('插入', int(i / bulk_num), '批数据')
                success, failed = bulk(self.es_client, load_data, index=self.index, raise_on_error=True)
                del load_data[0:len(load_data)]
                print(success, failed)

        if len(load_data) > 0:
            success, failed = bulk(self.es_client, load_data, index=self.index, raise_on_error=True)
            del load_data[0:len(load_data)]
            print(success, failed)


if __name__ == '__main__':
    from datetime import datetime
    es = ElasticSearchClient.get_es_servers()
    es.index(index='fo-qa', doc_type='test-type', body={'any': 'data', 'timestamp': datetime.now()})
    load_es = LoadElasticSearch()
    qa_list = []
    with open(cur_dir + '/qa_data/basic_data/fo_qa.txt', 'r') as f:
        for record in f.readlines():
            record_list = record.strip().split(',')
            try:
                qa_list.append({'topic': record[5], 'title': record[1], 'question': record[3], 'answer': record[4]})
            except:
                pass

    load_es.add_data_bulk(qa_list)  # 批量加载

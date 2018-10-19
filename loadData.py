#!/usr/bin/env python
# encoding: utf-8
from torchtext import data
import torch
import pickle

from config import VOCABULARY_SIZE, BATCH_SIZE

"""
导入数据
"""
TEXT = data.Field()
WEIBO_ID = data.Field()
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

train, val = data.TabularDataset.splits(
    path='./data/nationalism/', train='train_clean.csv',
    validation='dev_clean.csv', format='csv',
    fields=[('label', LABEL), ('text', TEXT)])

test = data.TabularDataset.splits(
    path='./data/nationalism/', test='test.csv',
    format='csv',
    fields=[('weibo_id', WEIBO_ID), ('text', TEXT)]
)

TEXT.build_vocab(train, max_size=VOCABULARY_SIZE)
LABEL.build_vocab(train)

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, val),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    repeat=True)

test_iterator = data.BucketIterator(test,
                                    batch_size=BATCH_SIZE,
                                    sort_key=lambda x: len(x.text),
                                    repeat=False)

print('finish load data')

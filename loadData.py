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
WEIBO_ID = data.Field(sequential=False, use_vocab=False)
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

train, val = data.TabularDataset.splits(
    path='./data/nationalism/', train='train_clean.csv',
    validation='dev_clean.csv', format='csv',
    fields=[('label', LABEL), ('text', TEXT)])

test = data.TabularDataset(
    path='./data/nationalism/test_clean.csv',
    format='csv',
    fields=[('weibo_id', WEIBO_ID), ('text', TEXT)]
)

TEXT.build_vocab(train, max_size=VOCABULARY_SIZE)
LABEL.build_vocab(train)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, val, test),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    repeat=True)

print('finish load data')
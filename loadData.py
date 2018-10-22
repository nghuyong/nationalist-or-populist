#!/usr/bin/env python
# encoding: utf-8
from torchtext import data
import torch

from config import VOCABULARY_SIZE, BATCH_SIZE, CURRENT_MODEL_NAME

TEXT = data.Field()
WEIBO_ID = data.Field(sequential=False, use_vocab=False)
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

train, val = data.TabularDataset.splits(
    path='./data/{}/'.format(CURRENT_MODEL_NAME), train='train_clean.csv',
    validation='dev_clean.csv', format='csv',
    fields=[('label', LABEL), ('text', TEXT)])

test = data.TabularDataset(
    path='./data/{}/test_clean.csv'.format(CURRENT_MODEL_NAME),
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

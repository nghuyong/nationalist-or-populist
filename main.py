#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.optim as optim
import torch.nn as nn
from model.RNN import RNN
from model.LSTM import LSTM
from torchtext import data

TEXT = data.Field()
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

train, val = data.TabularDataset.splits(
    path='./data/nationalism/', train='train_clean.csv',
    validation='dev_clean.csv', format='csv',
    fields=[('label', LABEL), ('text', TEXT)])

TEXT.build_vocab(train, max_size=10000)
LABEL.build_vocab(train)

BATCH_SIZE = 1

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, val),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    repeat=False)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda')

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    batch_count = 1
    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text.cuda()).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward(retain_graph=True)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if batch_count % 100 == 0:
            print(batch_count, epoch_acc / batch_count, epoch_loss / batch_count)

        batch_count += 1

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    print(
        f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%'
    )

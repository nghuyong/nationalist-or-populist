#!/usr/bin/env python
# encoding: utf-8
from pprint import pprint
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from config import INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, OUTPUT_DIM, BATCH_SIZE, LEARNING_RATE, CURRENT_MODEL_NAME
from loadData import train_iterator, valid_iterator, test_iterator
from model.LSTM import LSTM
import pandas as pd

max_acc = 0.0


def binary_accuracy(preds, y):
    """
    计算统计参数，正确率,精确率,召回率,f值
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))

    true_positive = (rounded_preds * y).sum(dim=0)

    precision = true_positive.div(rounded_preds.sum(dim=0).add(1e-9))
    recall = true_positive.div(y.sum(dim=0).add(1e-9))

    f_value = 1.0 / (1.0 / precision + 1.0 / recall)

    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc.item(), precision.item(), recall.item(), f_value.item()


def train(model, iterator, optimizer, criterion):
    global max_acc
    model.train()
    for batch_index, batch in tqdm(enumerate(iterator)):
        if len(batch.label) != BATCH_SIZE:
            break
        optimizer.zero_grad()
        predictions = model(batch.text.cuda()).squeeze(1)

        loss = criterion(predictions, batch.label)

        loss.backward(retain_graph=True)

        optimizer.step()

        if (batch_index + 1) % 100 == 0:
            train_acc, train_precision, train_recall, train_f_value = binary_accuracy(predictions, batch.label)
            dev_acc, dev_precision, dev_recall, dev_f_value, dev_loss = evaluate(model, valid_iterator, criterion)
            if dev_acc > max_acc:
                max_acc = dev_acc
                torch.save(model, './trainedModel/best_{}_model.pkl'.format(CURRENT_MODEL_NAME))
                print('save model', flush=True)
            print(
                """
                batch {}
                train acc {:.4f} precision {:.4f} recall {:.4f} f {:.4f} loss {:.4f}
                  dev acc {:.4f} precision {:.4f} recall {:.4f} f {:.4f} loss {:.4f}
                """.format(batch_index + 1, train_acc, train_precision, train_recall, train_f_value, loss.item(),
                           dev_acc, dev_precision, dev_recall, dev_f_value, dev_loss), flush=True
            )


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f_value = 0
    model.eval()
    all_batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(iterator):
            if len(batch.label) != BATCH_SIZE:
                all_batch_count = batch_index + 1
                break
            predictions = model(batch.text.cuda()).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc, precision, recall, f_value = binary_accuracy(predictions, batch.label)
            epoch_loss += loss
            epoch_acc += acc
            epoch_precision += precision
            epoch_recall += recall
            epoch_f_value += f_value
    model.train()
    return (epoch_acc / all_batch_count, epoch_precision / all_batch_count, epoch_recall / all_batch_count,
            epoch_f_value / all_batch_count, epoch_loss / all_batch_count)


def test(model, iterator):
    model.eval()
    zero_count = 0
    one_count = 0
    weibo_id_prediction_dic = dict()
    with torch.no_grad():
        for batch_index, batch in tqdm(enumerate(iterator)):
            if len(batch.weibo_id) != BATCH_SIZE:
                break
            predictions = model(batch.text.cuda()).squeeze(1)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            for index in range(BATCH_SIZE):
                if rounded_preds[index].item() < 1:
                    zero_count += 1
                else:
                    one_count += 1
                weibo_id_prediction_dic[batch.weibo_id[index].item()] = int(rounded_preds[index].item())
        print(
            f'positive num {one_count} rate {one_count * 1.0 / (one_count + zero_count):.3f} negative num {zero_count} rate {zero_count * 1.0 / (one_count + zero_count):.3f}')
    print('start to generate result excel')
    df = pd.read_excel('./data/sourceData/test.xlsx')
    new_df = df[['_id', '_id_x', '_id_y', 'nick_name', 'content']]
    nationalism_predictions = []
    for index, each in tqdm(new_df.iterrows()):
        nationalism_predictions.append(weibo_id_prediction_dic.get(int(each["_id"]), ""))
    new_df['{}_prediction'.format(CURRENT_MODEL_NAME)] = nationalism_predictions
    new_df.to_excel('./predictionResults/{}_prediction_result.xlsx'.format(CURRENT_MODEL_NAME))


if __name__ == "__main__":
    is_traning = False
    if is_traning:
        model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE)
    else:
        model = torch.load('./trainedModel/best_{}_model.pkl'.format(CURRENT_MODEL_NAME))
        print('load model successfully')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda')
    model = model.to(device)
    criterion = criterion.to(device)
    if is_traning:
        for i in range(5):
            train(model, train_iterator, optimizer, criterion)
    else:
        test(model, test_iterator)

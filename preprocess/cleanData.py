#!/usr/bin/env python
# encoding: utf-8
import re
import jieba


def clean(source_file, target_file):
    train_f = open(source_file, 'r', encoding='utf-8')
    sentences = train_f.readlines()

    train_clean_f = open(target_file, 'w', encoding='utf-8')
    jieba.suggest_freq('NUMBER')
    for line in sentences:
        line = line.strip()
        label = line[0]
        train_clean_f.write('{},'.format(label))
        line = line[1:]
        line = re.sub(r'h?ttp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                      line)  # 删除URL
        line = re.sub(r'\s+', '', line)  # 删除多余空格
        line = re.sub(r'】|【|\[|\]|\/|『 |』', '', line)  # 删除特殊的符号
        line = re.sub(r'\d+', 'NUMBER', line)  # 数字统一转换为 NUM
        train_clean_f.write(" ".join(jieba.cut(line)) + '\n')


if __name__ == "__main__":
    clean('../data/nationalism/dev.txt', '../data/nationalism/dev_clean.csv')
    clean('../data/nationalism/train.txt', '../data/nationalism/train_clean.csv')

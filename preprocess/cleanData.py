#!/usr/bin/env python
# encoding: utf-8
import re
import jieba

url_re = re.compile(r'h?ttp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
white_blank_re = re.compile(r'\s+')
keyword_re = re.compile(r'转基因')
special_character_re = re.compile(r'】|【|\[|\]|\/|『 |』')
number_re = re.compile(r'\d+')


def clean_weibo_text(weibo_text):
    weibo_text = re.sub(url_re, '', weibo_text)  # 删除URL
    weibo_text = re.sub(keyword_re, '', weibo_text)  # 删除核心单词
    weibo_text = re.sub(white_blank_re, '', weibo_text)  # 删除多余空格
    weibo_text = re.sub(special_character_re, '', weibo_text)  # 删除特殊的符号
    weibo_text = re.sub(number_re, 'NUMBER', weibo_text)  # 数字统一转换为 NUM
    return " ".join(jieba.cut(weibo_text)) + '\n'


def clean(source_file, target_file):
    source_f = open(source_file, 'r', encoding='utf-8')
    sentences = source_f.readlines()
    clean_f = open(target_file, 'w', encoding='utf-8')
    jieba.suggest_freq('NUMBER')
    if 'test' in source_file:
        for line in sentences:
            line = line.strip()
            if not line:
                continue
            try:
                weibo_id, weibo_text = line.split(',', 1)
            except:
                continue
            clean_f.write('{},'.format(weibo_id))
            clean_f.write(clean_weibo_text(weibo_text))
    else:
        for line in sentences:
            line = line.strip()
            label = line[0]
            clean_f.write('{},'.format(label))
            weibo_text = line[1:]
            clean_f.write(clean_weibo_text(weibo_text))


if __name__ == "__main__":
    clean('../data/populism/dev.txt', '../data/populism/dev_clean.csv')
    clean('../data/populism/train.txt', '../data/populism/train_clean.csv')
    clean('../data/populism/test.csv', '../data/populism/test_clean.csv')

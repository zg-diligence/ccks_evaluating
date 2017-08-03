#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
files_size = 300
reload(sys)
sys.setdefaultencoding('utf-8')

# 拆分文件中的句子
def separate_file(texts):
    sep_texts, numbers = [], []
    puncs = '！？。'
    for text in texts:
        k, sentence_num = 0, 0
        while k < len(text):
            if text[k][0].encode('utf8') in puncs:
                sep_texts.append(text[:k+1])
                text, k = text[k+1:], 0
                sentence_num += 1
            else:
                k += 1
        if len(text) > 0:
            sep_texts.append(text)
            sentence_num += 1
        numbers.append(sentence_num)
    return sep_texts, numbers

path = os.getcwd()[:-5].replace('\\', '/') + '/ccks/WI_DEV/test_crfsuite'

folders = list(os.listdir(path))    # 获取文件夹根目录
func = lambda name: int(name[:-11]) # 文件名处理函数
read_text = lambda filepath: [line.strip().split('\t') for line in open(filepath)] # 读取单个文件
files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders] # 获取所有文件名
all_texts = [read_text(path + '/' + folders[k] + '/' + one_file) for k in range(4) for one_file in files[k]]
all_texts.pop(60)
all_sents, sents_num = separate_file(all_texts)
print sum(sents_num)
w = open('WI_Test.txt', 'w')

for sent in all_sents:
    for line in sent:
        w.write('\t'.join(line) + '\n')
    w.write('\n')
w.close()
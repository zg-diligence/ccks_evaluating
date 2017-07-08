#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	function: 半监督训练
	NOTE: linux系统下运行
"""

import os
import time
import sklearn
import pycrfsuite
import itertools

files_size = 300

# 读取训练和测试语料
def read_texts(path):
    folders = list(os.listdir(path)) 	# 获取文件夹根目录
    func = lambda name: int(name[:-11]) # 文件名处理函数
    read_text = lambda filepath: [line.strip().split('\t') for line in open(filepath)] 		# 读取单个文件
    files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders]   # 获取所有文件名
    return [read_text(path + '/' + folders[k] + '/' + one_file) for k in range(4) for one_file in files[k][:100]]

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

# 获取单句attributes
def sent2attributes(sent, templates):
    # 获取单字单模板attribute
    def get_attribute(sent, i, template, flag):
        attr = ''
        if 'word' == flag:
            for k in template:
                if i + k < 0:
                    attr += '|' + str(k) + ':BOS'
                elif i + k > len(sent) - 1:
                    attr += '|' + str(k) + ':EOS'
                else:
                    attr += '|' + str(k) + ':' + sent[i + k][0]
        else:
            for k in template:
                if i + k < 0:
                    attr += '|' + str(k) + ':BOS'
                elif i + k > len(sent) - 1:
                    attr += '|' + str(k) + ':EOS'
                else:
                    attr += '|' + str(k) + ':' + sent[i + k][1]
        return  attr[1:]

    # 获取单字attributes
    def word2attributes(sent, templates, i):
        attrs = []
        for template in templates[0]:
            attrs.append(get_attribute(sent, i, template, 'word'))
        for template in templates[1]:
            attrs.append(get_attribute(sent, i, template, 'pos'))
        return attrs

    return [word2attributes(sent, templates, i) for i in range(len(sent))]

# 获取单句标签序列
def sent2labels(sent):
    return [label for token, postag, label in sent]

# 单次半监督训练
def crfsuite_main(train_sents, extra_sents, templates, crfsuite_model):
    # 构造训练集、测试集特征及对应label
    x_train = [sent2attributes(s, templates) for s in train_sents + extra_sents]
    y_train = [sent2labels(s) for s in train_sents + extra_sents]

    # 创建训练器,加载训练集,选择算法,设置训练参数,训练模型
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train): trainer.append(xseq, yseq)
    trainer.select('pa')  # 例如: 'lbfgs','l2sgd','ap','pa','arow'
    trainer.set_params({
        'max_iterations': 17,
        'feature.possible_transitions': True})
    trainer.train(crfsuite_model)
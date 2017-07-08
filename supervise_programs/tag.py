#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用已经训练好的模型标注测试集
"""

import os
import pycrfsuite

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

files_size = 100

template3 = (
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)),
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)))

def read_texts(path):
    folders = list(os.listdir(path))    # 获取文件夹根目录
    func = lambda name: int(name[:-11]) # 文件名处理函数
    read_text = lambda filepath: [line.strip().split('\t') for line in open(filepath)]      # 读取单个文件
    files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders]   # 获取所有文件名
    return [read_text(path + '/' + folders[k] + '/' + one_file) for k in range(4) for one_file in files[k]]

def write_preds(path, texts, preds):
    def write_pred(path, text, pred):
        w = open(path, 'w')
        for line in text:
            w.write('\t'.join(line) + '\t' + pred.pop(0) + '\n')
        w.close()

    folders = list(os.listdir(path))
    func = lambda name: int(name[:-11])
    files = [sorted(list(os.listdir(path + '/' + folder)), key=func) for folder in folders]
    for i in range(4):
        for j in range(files_size):
            write_pred(path + '/' + folders[i] + '/' + files[i][j],
                       texts[i * files_size + j], preds[i * files_size + j][:])

def sent2attributes(sent, templates):
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

    def word2attributes(sent, templates, i):
        attrs = []
        for template in templates[0]:
            attrs.append(get_attribute(sent, i, template, 'word'))
        for template in templates[1]:
            attrs.append(get_attribute(sent, i, template, 'pos'))
        return attrs

    return [word2attributes(sent, templates, i) for i in range(len(sent))]

def tag_texts(texts, crfsuite_model, templates):
    x_test = [sent2attributes(text, templates) for text in texts]
    tagger = pycrfsuite.Tagger()
    tagger.open(crfsuite_model)
    return [tagger.tag(xseq) for xseq in x_test]

if __name__ == '__main__':
    path = os.getcwd()[:-5].replace('\\','/') + "/test_crfsuite"
    texts = read_texts(path)
    preds = tag_texts(texts, 'total2.crfsuite', template3)
    write_preds(path, texts, preds)
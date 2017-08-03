#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  train-data全部用于训练,然后测试test-data
"""

import os
import time
import pycrfsuite

files_size = 300

# 读取训练和测试语料
def read_texts(path):
    """
    @grams:
        path:四个文件夹的根目录
    @return:
        list, 四个文件中所有的文本
    """

    folders = list(os.listdir(path))  # 获取文件夹根目录
    func = lambda name: int(name[:-11])  # 文件名处理函数
    read_text = lambda filepath: [line.strip().split('\t') for line in open(filepath)]  # 读取单个文件
    files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders]  # 获取所有文件名
    return [read_text(path + '/' + folders[k] + '/' + one_file) for k in range(4) for one_file in files[k]]

# 拆分文件中的句子
def separate_file(texts):
    """
    @grams:
        texts: list, 文本列表
    @return:
        sep_texts: list, 拆分后的句子列表
        numbers: list, 文本拆分后的句子数目
    """

    sep_texts, numbers = [], []
    puncs = '！？。'
    for text in texts:
        k, sentence_num = 0, 0
        while k < len(text):
            if text[k][0] in puncs:
                sep_texts.append(text[:k + 1])
                text, k = text[k + 1:], 0
                sentence_num += 1
            else:
                k += 1
        if len(text) > 0:
            sep_texts.append(text)
            sentence_num += 1
        numbers.append(sentence_num)
    return sep_texts, numbers

# 合并文件中的句子
def union_file(file_num, preds):
    """
    @grams:
        file_num:list, 文本拆分后句子的数目
        preds:list, 句子预测结果
    @return:
        union_preds:list, 文本的预测结果
    """

    union_preds = []
    for num in file_num:
        pred = []
        for i in range(num):
            pred += preds[i]
        preds = preds[num:]
        union_preds.append(pred)
    return union_preds

# 提取单句特征
def sent2attributes(sent, templates):
    """
    @grams:
        sent:list, 单个句子, 单字格式list, word|pos|label
    @return:
        list, 单个句子特征
    """

    # 提取单字单模板特征
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
        return attr[1:]

    # 提取单字所有特征
    def word2attributes(sent, templates, i):
        attrs = []
        for template in templates[0]:  # 字特征
            attrs.append(get_attribute(sent, i, template, 'word'))
        for template in templates[1]:  # 分词和词性特征
            attrs.append(get_attribute(sent, i, template, 'pos'))
        return attrs

    return [word2attributes(sent, templates, i) for i in range(len(sent))]

# 提取单句标签序列
def sent2labels(sent):
    return [label for token, postag, label in sent]

# 有监督训练
def supervise_train(train_sents, templates, max_iter):
    print '有监督训练开始!', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    x_train = [sent2attributes(s, templates) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train): trainer.append(xseq, yseq)
    trainer.select('pa')
    trainer.set_params({
        'max_iterations': max_iter,
        'epsilon': 1e-5})
    trainer.train('ccks.crfsuite')

    print '有监督训练结束!', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# 标注测试集并评测
def tag_evaluate(test_sents, crfsuite_model, templates):
    x_test = [sent2attributes(s, templates) for s in test_sents]
    y_true = [sent2labels(s) for s in test_sents]
    tagger = pycrfsuite.Tagger()
    tagger.open(crfsuite_model)
    y_pred = [tagger.tag(xseq) for xseq in x_test]

    # 预测结果写入文本
    w = open('result.txt', 'w')
    for sents, pred in zip(test_sents, y_pred):
        for i in range(len(sents)):
            w.write('\t'.join([sents[i][0], sents[i][1], sents[i][2], pred[i]]) + '\n')
    w.close()

    # 评测并将结果写入文本
    print '\n实体评测结果:'
    os.system('perl conlleval.pl -d "\t" < result.txt')
    report = os.popen('perl conlleval.pl -d "\t" < result.txt').readlines()
    report = [line.encode('utf8') for line in report]
    open('supervise_report.txt', 'a').writelines(report)

# 全部有监督训练
def supervise_main(train_path, test_path, templates, max_iter):
    # 读取训练语料
    train_texts = read_texts(train_path)
    train_texts_4 = [train_texts[i * files_size: (i + 1) * files_size] for i in range(4)]

    # 调整四个文件夹下文本的分布并分句
    dev_texts_4 = [[] for i in range(4)]
    for i in range(4):
        for j in range(files_size // 4):
            for k in range(4):
                dev_texts_4[i].append(train_texts_4[k].pop(0))
    train_texts_4 = dev_texts_4
    train_sents = []
    for texts in train_texts_4: train_sents += separate_file(texts)[0]
    supervise_train(train_sents, templates, max_iter)

    test_texts = read_texts(test_path)
    test_sents = separate_file(test_texts)[0]
    tag_evaluate(test_sents, 'ccks.crfsuite', templates)

if __name__ == '__main__':
    template = (
        ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)),
        ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)))

    path = os.getcwd()[:-5].replace('\\','/')
    src_path = path + '/ccks/SCIR/train_crfsuite'
    dest_path = path + '/ccks/SCIR/test_crfsuite'

    supervise_main(src_path, dest_path, template, 17)
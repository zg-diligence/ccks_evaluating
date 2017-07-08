#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    function: 有监督训练&留一交叉法
    NOTE: linux系统下运行
"""

import os
import time
import copy
import random
import nltk
import sklearn
import pycrfsuite
import multiprocessing

files_size = 300 # 单个文件夹下文件数目,共4个文件夹

# 读取训练和测试语料
def read_texts(path):
    """
    @grams:
        path:四个文件夹的根目录
    @return:
        list, 四个文件中所有的文本
    """

    folders = list(os.listdir(path))    # 获取文件夹根目录
    func = lambda name: int(name[:-11]) # 文件名处理函数
    read_text = lambda filepath: [line.strip().split('\t') for line in open(filepath)]    # 读取单个文件
    files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders] # 获取所有文件名
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

# 将预测结果写入文本 -- 4个文件夹&单个文件、整合文件&用于评测
def write_preds(src_path, dest_path, pred, k):
    """
    @grams:
        src_path:预处理结果文件夹根目录 格式word|pod|label
        dest_path:预测结果文件根目录 格式word|pos|label|pred
        pred:list, 文本预测结果
        k:测试集文本数目
    @return: null
    """

    # 调整预测结果文件顺序
    def adjust_pred(pred, k):
        res = []
        for i in range(4):
            for m in range(files_size // k):
                res += pred[4*m*k + i*k: 4*m*k + (i+1)*k]
        return res

    # 写入单个文件预测结果
    def write_pred(src_path, dest_path, info):
        src_w = open(src_path)
        dest_w = open(dest_path, 'w')
        for line in src_w:
            dest_w.write(line.strip() + '\t' + info[0] + '\n')
            all_res.write(line.strip() + '\t' + info.pop(0) + '\n')
        src_w.close()
        dest_w.close()

    adj_pred = adjust_pred(pred, k)               # 调整预测结果文件顺序
    src_func = lambda name: int(name[:- 11])
    src_folders = list(os.listdir(src_path))
    src_files = [sorted(list(os.listdir(src_path + '/' + files)), key = src_func) for files in src_folders]
    all_res = open('supervise_sents.txt', 'w')    # 所有文本预测结果写入同一个文件, 用于评测
    for i in range(4):
        for j in range(len(src_files[i])):
            src_file = src_path + '/' + src_folders[i] + '/' + src_files[i][j]
            dest_file = dest_path + '/' + src_folders[i] + '/' + str(j+1) + '.txt'
            write_pred(src_file, dest_file, adj_pred[i * files_size + j])
    all_res.close()

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
        return  attr[1:]

    # 提取单字所有特征
    def word2attributes(sent, templates, i):
        attrs = []
        for template in templates[0]: # 字特征
            attrs.append(get_attribute(sent, i, template, 'word'))
        for template in templates[1]: # 分词和词性特征
            attrs.append(get_attribute(sent, i, template, 'pos'))
        return attrs

    return [word2attributes(sent, templates, i) for i in range(len(sent))]

# 提取单句标签序列
def sent2labels(sent):
    return [label for token, postag, label in sent]

# 打印训得模型特性信息
def print_features(info, path):
    # 打印状态转换信息
    def print_transitions(trans_features, w):
        for (label_from, label_to), weight in trans_features:
            w.write("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
            w.write('\n')

    # 打印状态信息
    def print_state_features(state_features, w):
        for (attr, label), weight in state_features:
            w.write("%0.6f %-6s %s" % (weight, label, attr))
            w.write('\n')

    w = open(path, 'w')

    # 打印状态转换信息
    w.write('Top likely transitions:\n')
    print_transitions(Counter(info.transitions).most_common(100), w)
    w.write('\nTop unlikely transitions:\n')
    print_transitions(Counter(info.transitions).most_common()[-100:], w)

    # 打印状态信息
    w.write('\nTop positive:\n')
    print_state_features(Counter(info.state_features).most_common(100), w)
    w.write('\nTop negative:\n')
    print_state_features(Counter(info.state_features).most_common()[-100:], w)
    w.close()

# 判断是否是特殊字符
def isSpecial(character):
    punc = '[!"#$%&\'()*+,-./:;<=>?@\\^_`{}~]|！￥……（）—；‘’、【】《》？“：，”★≥'
    if character in punc:
        return 'S'
    if character.isdigit():
        return 'D'
    if character.isalpha():
        return 'C'
    return 'O'
 
# 返回BIE分项评测结果
def bio_classification_report(y_true, y_pred):
    """
    @grams:
        y_true:正确标注序列
        y_pred:预测结果序列
    @return:
        BIE分项评测结果
    """

    lb = sklearn.preprocessing.LabelBinarizer()
    y_true_combined = lb.fit_transform(list(itertools.chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(itertools.chain.from_iterable(y_pred)))
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    return sklearn.metrics.classification_report(y_true_combined, y_pred_combined,
                  labels=[class_indices[cls] for cls in tagset], target_names=tagset, digits = 4)

# 打印评测报告 
def print_report():
    """
    @note:
        supervise_sents.txt:预测结果整合文本
    """

    print '\n实体评测结果:'
    os.system('perl conlleval.pl -d "\t" < supervise_sents.txt')

# 将提取的句子特征写入文件
def print_attributes(sent):
    w = open('features.txt','w')
    for feature in sent:
        for word in feature:
            w.write(str(word + '\t\t').encode('utf-8'))
        w.write('\n')
    w.close()

# 单次有监督训练
def supervise_train(train_sents, templates, crfsuite_model, k, max_iter):
    """
    @grams:
        train_sents:list, 用于训练的所有句子
        crfsuite_model:训练所得模型存储路径
        k:语料拆分为k份
        max_iter:模型训练最大迭代次数
    """

    print 'model -', k, '训练开始'
    # 构造训练集、测试集特征及对应label
    x_train = [sent2attributes(s, templates) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    # 创建训练器,加载训练集,选择算法,设置训练参数,训练模型
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train): trainer.append(xseq, yseq)
    trainer.select('pa')  # 例如: 'lbfgs','l2sgd','ap','pa','arow'
    trainer.set_params({
        'max_iterations': max_iter,
         'epsilon': 1e-5})
    trainer.train(crfsuite_model)
    print 'model -', k, '训练完成'

# 全部有监督训练
def supervise_trains(sents_4, templates, test_k, max_iter):
    """
    @grams:
        sents_4:四个文件夹下所有的句子, 用四个list存储
        test_k:测试集文本个数
        max_iter:模型训练最大迭代次数
    """

    print '有监督训练开始!', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    length_4 = [len(sents) // (files_size // test_k) for sents in sents_4]

    # 多进程同时训练, 无返回值
    threads = []
    for k in range(files_size // test_k):
        train_sents = []
        for m in range(4):
            train_sents += sents_4[m][:k*length_4[m]] + sents_4[m][(k+1)*length_4[m]:]
        function_args = (train_sents, templates, str(k) + '.crfsuite', k, max_iter)
        threads.append(multiprocessing.Process(target=supervise_train, args=function_args))

    # 所有语料训练最终模型
    all_sents = []
    for i in range(4):
        all_sents += sents_4[i]
    function_args = (all_sents, templates, 'total2.crfsuite', 17, max_iter)
    threads.append(multiprocessing.Process(target=supervise_train, args=function_args))

    for t in threads: t.start()
    for t in threads: t.join()
    print '有监督训练结束!', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# 标注测试集并评测
def tag_evaluate(test_sents, crfsuite_model, templates):
    """
    @grams:
        test_sents:list, 测试集所有句子
        crfsuite_model:已训得模型的路径
    @return:
        y_true:list, 所有句子的正确标注序列
        y_pred:list, 所有句子的预测结果序列
    """

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
    report = os.popen('perl conlleval.pl -d "\t" < result.txt').readlines()
    report = [line.encode('utf8') for line in report]
    open('supervise_report.txt', 'a').writelines(report)
    return y_true, y_pred

# 全部有监督训练
def supervise_main(src_path, dest_path, test_k, templates, max_iter):
    """
    @grams:
        src_path:预处理结果文件夹根目录
        dest_path:预测结果文件夹根目录
        test_k:训练集文本数目
        max_iter:有监督训练最大迭代次数
    """

    # 读取训练和测试语料
    all_texts = read_texts(src_path)  
    texts_4 = [all_texts[i * files_size: (i + 1) * files_size] for i in range(4)]
    
    # 调整四个文件夹下文本的分布
    dev_texts_4 = [[] for i in range(4)]
    for i in range(4):
        for j in range(files_size//4):
            for k in range(4):
                dev_texts_4[i].append(texts_4[k].pop(0))
    texts_4 = dev_texts_4

    # 分句
    sents_4 = [separate_file(texts)[0] for texts in texts_4]
    length_4 = [len(sents) // (files_size // test_k) for sents in sents_4]

    # 有监督训练得到10个初始模型
    supervise_trains(sents_4, templates, test_k, max_iter)     

    # 用已训得模型标注对应测试集
    open('supervise_report.txt', 'w')
     y_true, y_pred, all_test_sents = [], [], []
    for k in range(files_size//test_k):
        test_sents = []
        for m in range(4):
            test_sents += sents_4[m][k*length_4[m]:(k+1)*length_4[m]]
        true, pred = tag_evaluate(test_sents, str(k)+'.crfsuite', templates)
        all_test_sents += test_sents
        y_true += true
        y_pred += pred

    # 将所有预测结果写入文件, 评测
    w = open('supervise_sents.txt', 'w')
    for sent, y_pred in zip(all_test_sents, y_pred):
        for i in range(len(sent)):
            line = sent[i] + [y_pred[i]]
            w.write('\t'.join(line) + '\n')
    w.close()
    print_report()

    # 将所有预测结果附加在预处理结果后面,并写入文件
    # write_preds(src_path, dest_path, y_pred, test_k) # 前提是没有调整文本分布
    
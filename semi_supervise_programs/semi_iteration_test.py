#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import math
import pycrfsuite
import semi_iteration as ccks
import multiprocessing

files_size = 300

template3 = (
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)),
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)))

# 所有实体类别
entity = ('B-body', 'I-body', 'E-body', 'B-chec', 'I-chec', 'E-chec',
          'B-symp', 'I-symp', 'E-symp', 'B-dise', 'I-dise', 'E-dise',
          'B-cure', 'I-cure', 'E-cure', 'O')

# 计算句子的平均信息熵
def compute_entropy(attr, tagger):
    """
    @grams:
        attr:list, 句子特征序列
        tagger:标注器
    @return:
        句子的平均信息熵
    """

    probs = []
    tagger.set(attr[1])
    for pos in range(len(attr[1])):
        try:
            prob = [tagger.marginal(item, pos) for item in entity]
            entropy = 0
            for p in prob:
                entropy += p * math.log(1 / p)
            probs.append(entropy)
        except ValueError:
            print len(attr[1]), pos
    return sum(probs) / len(attr[1])

# 返回文件夹下所有句子的平均信息熵
def compute_entropys(attrs, tagger):
    return [compute_entropy(attr, tagger) for attr in attrs]

# 选择信息熵最小的k个句子 k-max-heap
def select_sents(sents, entropys, k):
    def push_down(sents_entropys, i):
        pos, sent_entropy = i, sents_entropys[i]
        while pos * 2 < len(sents_entropys):
            child = pos * 2
            if child + 1 < len(sents_entropys) and \
                            sents_entropys[child + 1][1] > sents_entropys[child][1]:
                child += 1
            if sents_entropys[child][1] <= sent_entropy[1]:
                break
            else:
                sents_entropys[pos], pos = sents_entropys[child], child
        sents_entropys[pos] = sent_entropy

    def build_max_heap(sents_entropys):
        for pos in range(len(sents_entropys) // 2, 0, -1):
            push_down(sents_entropys, pos)
        return sents_entropys

    sents_entropys = zip(sents, entropys)
    k_heap = build_max_heap(sents_entropys[:k+1])
    pos, sents_entropys = 0, sents_entropys[k+1:]
    length = len(sents_entropys)
    for pos in range(length):
        first = sents_entropys.pop(0)
        if k_heap[1][1] > first[1]:
            sents_entropys.append(k_heap[1])
            k_heap[1] = first
            push_down(k_heap, 1)
        else:
            sents_entropys.append(first)
    return [sent for sent, entropy in k_heap[1:]], [sent for sent, entropy in sents_entropys]

# 标注测试集并评测
def tag_evaluate(test_sents, crfsuite_model, templates, k):
    x_test = [ccks.sent2attributes(s, templates) for s in test_sents]
    tagger = pycrfsuite.Tagger()
    tagger.open(crfsuite_model)
    y_pred = [tagger.tag(xseq) for xseq in x_test]

    # 将单次迭代后的预测结果写入文件并评测
    # w = open('result.txt', 'w')
    # for sents, pred in zip(test_sents, y_pred):
    #     for i in range(len(sents)):
    #         w.write('\t'.join([sents[i][0], sents[i][1], sents[i][2], pred[i]]) + '\n')
    # w.close()
    # report = os.popen('perl conlleval.pl -d "\t" < result.txt').readlines()
    # report = [line.encode('utf8') for line in report]
    # open(str(k+1) + '-report.txt', 'a').writelines(report)

    return y_pred

# 用上一轮训练的模型重新标注测试集
def re_tag(sents_attrs_labels_4, crfsuite_model):
    tagger = pycrfsuite.Tagger()
    tagger.open(crfsuite_model)
    tmp_sals_4 = []
    for i in range(4):
        sents = [sent for sent, attr, label in sents_attrs_labels_4[i]]
        attrs = [attr for sent, attr, label in sents_attrs_labels_4[i]]
        labels = [tagger.tag(attr) for attr in attrs]
        tmp_sals_4.append([([], [], [])] + zip(sents, attrs, labels))
    return tmp_sals_4, tagger

# 迭代半监督训练
def train_process(train_sents, test_sents, semi_sents_attrs, model_path, k, templates):
    """
    @grams:
        train_sents:有监督训练语料训练集
        test_sents:有监督训练语料测试集
        semi_sents_attrs:半监督训练语料
        model_path:存储训得模型的文件夹路径
    @return:
        y_pred:最后一次迭代后的预测结果
    """

    # 基于有监督已训得模型标注半监督语料
    crfsuite_model = str(k) + '.crfsuite'
    tagger = pycrfsuite.Tagger()
    tagger.open(crfsuite_model)
    sents_attrs_labels_4 = []
    semi_sents, semi_attrs = semi_sents_attrs
    for i in range(4):
        labels = [tagger.tag(attr) for attr in semi_attrs[i]]          # 标注单个文件夹句子
        sents_attrs_labels_4.append([([], [], [])] + zip(semi_sents[i], semi_attrs[i], labels))  # 句子、特征、实体配对
    k_values = [len(sents_attrs_labels_4[i]) // 10 for i in range(4)]  # 确定单轮单个文件夹添加句子量K

    # 10次半监督迭代训练
    crfsuite_model = '0.crfsuite'
    selected_semi_sents, y_pred = [], []
    open(str(k+1)+'-report.txt', 'w').close()
    for p in range(1): # 1代表迭代一次,只加入1/10的半监督语料
        print '-'*20, 'k =', k+1, 'p =', p+1, '-'*20
        entropys, tmp_sents, selected_sals, left_sals = [], [], [], []
        for i in range(4):
            entropys = [0] + compute_entropys(sents_attrs_labels_4[i][1:], tagger)                  # 计算单文件夹句子的信息熵
            selected_and_not = select_sents(sents_attrs_labels_4[i], entropys, k_values[i])         # 选择单文件夹信息熵最小的K个句子
            selected_sals += selected_and_not[0]                                                    # K个信息熵最小的句子
            left_sals.append(selected_and_not[1])                                                   # 剩下的句子

        # 更改模型名称
        tmp = list(crfsuite_model)
        tmp[:-9] = str(int(''.join(tmp[:-9])) + 1)
        crfsuite_model = ''.join(tmp)

        selected_semi_sents += [[[sent[i][0],sent[i][1],label[i]] for i in range(len(sent))] for sent, attr, label in selected_sals]
        ccks.crfsuite_main(train_sents, selected_semi_sents, templates, model_path + crfsuite_model)# 重新训练模型
        y_pred = tag_evaluate(test_sents, model_path + crfsuite_model, templates, k)                # 标注测试集并评测
        sents_attrs_labels_4, tagger = re_tag(left_sals, model_path + crfsuite_model)               # 用新模型重新标注
    return y_pred

# 半监督训练, 基于有监督训练得到的模型分别进行半监督迭代训练
def semi_train_process_10(src_path, semi_path, model_path, templates, test_k):
    """
    @grams:
        src_path:有监督语料预处理结果文件夹根目录
        semi_path:半监督语料预处理结果文件夹根目录
        model_path:存储训得模型的文件夹路径
        test_k:测试集文本个数
    @return:
        y_pred:最后一次迭代后的预测结果
    """

    # 读取有监督训练语料并分句
    all_texts = ccks.read_texts(src_path)
    texts_4 = [all_texts[i * files_size: (i + 1) * files_size] for i in range(4)]
    sents_4 = [ccks.separate_file(texts)[0] for texts in texts_4]
    length_4 = [len(sents) // (files_size // test_k) for sents in sents_4]

    # 读取半监督训练语料并分文件夹处理
    semi_texts = ccks.read_texts(semi_path)
    texts_number = len(semi_texts) // 4
    semi_sents_4, semi_attrs_4 = [], [] # 单字格式word|pos|label
    for i in range(4):
        texts = semi_texts[i * texts_number:(i + 1) * texts_number]
        sents = ccks.separate_file(texts)[0]
        attrs = [ccks.sent2attributes(sent, templates) for sent in sents]
        semi_sents_4.append(sents)
        semi_attrs_4.append(attrs)

    # 基于有监督训练得到的模型分别进行半监督迭代训练 & 多进程,有返回值
    results, all_test_sents = [], []
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    for k in range(files_size//test_k):
        print '-'*20, '第', k + 1, '组', '-'*20
        train_sents, test_sents = [], []
        for m in range(4):
            train_sents += sents_4[m][:k*length_4[m]] + sents_4[m][(k+1)*length_4[m]:]
            test_sents += sents_4[m][k*length_4[m]:(k+1)*length_4[m]]
        all_test_sents += test_sents
        func_args = (train_sents, test_sents, (semi_sents_4, semi_attrs_4), model_path+str(k+1)+'-', k, templates)
        results.append(pool.apply_async(train_process, args=func_args))
    pool.close(); pool.join()

    # 获取半监督迭代训练最后的预测结果
    y_pred = []
    for t in results: 
        y_pred += t.get()

    # 将预测结果写入文件并评测
    w = open('semi_supervise_sents.txt', 'w')
    for sent, y_pred in zip(all_test_sents, y_pred):
        for i in range(len(sent)):
            line = sent[i] + [y_pred[i]]
            w.write('\t'.join(line) + '\n')
    w.close()
    os.system('perl conlleval.pl -d "\t" < semi_supervise_sents.txt')

if __name__ == '__main__':
    path = os.getcwd()[:-5].replace('\\','/')
    src_path = path + '/to_crfsuite'
    semi_path = path + '/semi_crfsuite'
    model_path = path + '/models/'

    semi_train_process_10(src_path, semi_path, model_path, template3, 30)
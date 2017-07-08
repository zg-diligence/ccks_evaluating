#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
有监督训练语料预处理文件,转换为crfsuite标准格式
NOTE:linux系统下运行
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import time

files_size = 300
dict_flag = False

# 将分词结果转换为单字格式
def pos2word(un_str):
    tempdata = un_str.decode('utf8').split('#')
    result = tempdata[0][0] + '\t' + tempdata[1] + 'B\n'
    for i in range(1, len(tempdata[0])):
        result += tempdata[0][i] + '\t' + tempdata[1] + 'I\n'
    return result

# 将数据转换成CRFsuite标准格式
def all_to_crfsuite(all_files):
    for k in range(len(pos_folders)):
        for ori_file, pos_file, ner_file in all_files[k]:
            # 将分词结果转换为以行存储 "word'\t'pos"
            path1 = crf_path + '/' + crf_folders[k] + '/' + pos_file[5:len(pos_file) - 7]
            w1 = open(pos_path + '/' + pos_folders[k] + '/' + pos_file)
            w2 = open(path1 + '-result1.txt','w') # 中间文件
            for line in w1:
                word_list = line.strip().split(' ')
                for i in range(len(word_list)):
                    w2.write(pos2word(word_list[i]))
            w1.close()
            w2.close()

            # 获取命名实体的首尾下标
            start, end = [[],[],[],[],[]], [[],[],[],[],[]]
            w3 = open(ner_path + '/' + ner_folders[k] + '/' + ner_file)
            for line in w3:
                temp = line.strip().decode('utf-8').split('\t')
                for n in range(5):
                    if temp[-1] == names_chs[n]:
                        start[n].append(int(temp[1]))
                        end[n].append(int(temp[2]))
            for n in range(5):
                start[n].append(0)
                end[n].append(0)
            w3.close()

            # 以BIOE形式标注字对应的实体部分
            i, j, mark, flag = 0, [0] * 5, [False] * 5, False
            w5 = open(path1 + '-result.txt', 'w')
            res_text = open(path1 + '-result1.txt').readlines()
            ori_text = open(ori_path + '/' + ori_folders[k] + '/' + ori_file).read().strip().decode('utf8')
            for tt in range(len(ori_text)):
                if not ori_text[tt].strip(): continue
                for n in range(5):
                    if tt == start[n][j[n]] and start[n][j[n]] != 0:
                        w5.write(res_text[i].strip() + '\tB-' + names_env[n] + '\n')
                        if tt != end[n][j[n]]:
                            mark[n] = True
                        elif end[n][0] != 0:
                            j[n] += 1
                        flag = True; break
                    elif mark[n]:
                        if tt == end[n][j[n]]:
                            w5.write(res_text[i].strip() + '\tE-' + names_env[n] + '\n')
                            mark[n], j[n] = False, j[n] + 1
                        else:
                            w5.write(res_text[i].strip() + '\tI-' + names_env[n] + '\n')
                        flag = True; break
                if not flag:
                    w5.write(res_text[i].strip() + '\tO\n')
                else:
                    flag = False
                i += 1
            w5.close()

            # 删除中间文件
            os.remove(path1 + '-result1.txt')

# 将数据转换成CRFsuite标准格式
def one_to_crfsuite(all_files, entity):
    for k in range(len(pos_folders)):
    	# if k == 1:break
        for ori_file, pos_file, ner_file in all_files[k]:
            # 将分词结果转换为以行存储 "word'\t'pos"
            path1 = crf_path + '/' + pos_folders[k][:2] + '/' + pos_file[5:len(pos_file) - 7]
            w1 = open(pos_path + '/' + pos_folders[k] + '/' + pos_file)
            w2 = open(path1 + '-result1.txt', 'w')  # 中间文件
            for line in w1:
                word_list = line.strip().split(' ')
                for i in range(len(word_list)):
                    w2.write(pos2word(word_list[i]))
            w1.close()
            w2.close()

            # 获取命名实体的首尾下标
            start, end = [], []
            w3 = open(ner_path + '/' + ner_folders[k] + '/' + ner_file)
            for line in w3:
                temp = line.strip().decode('utf-8').split('\t')
                if temp[-1] == entity:
                    start.append(int(temp[1]))
                    end.append(int(temp[2]))
            start.append(0)
            end.append(0)
            w3.close()

            # 以BIOE形式标注字对应的实体部分
            i, j, mark = 0, 0, False
            w5 = open(path1 + '-result.txt', 'w')
            res_text = open(path1 + '-result1.txt').readlines()
            ori_text = open(ori_path + '/' + ori_folders[k] + '/' + ori_file).read().strip().decode('utf8')
            for tt in range(len(ori_text)):
                if not ori_text[tt].strip():continue
                if tt == start[j] and start[0] != 0:
                    w5.write(res_text[i].strip() + '\tB-cure\n')
                    if tt != end[j]:
                        mark = True
                    elif start[0] != 0:
                        j += 1
                elif mark:
                    if tt == end[j]:
                        w5.write(res_text[i].strip() + '\tE-cure\n')
                        mark, j = False, j + 1
                    else:
                        w5.write(res_text[i].strip() + '\tI-cure\n')
                else:
                	w5.write(res_text[i].strip() + '\tO\n')
                i += 1
            w5.close()

            # 删除中间文件
            os.remove(path1 + '-result1.txt')

# 添加字典特征 -- 最大逆向匹配算法
def add_dict(k = 60):
	"""
	@grams:
		k:训练集文本数目
	"""

    def get_k_words(text, i, k):
        if i + 1 < k:
            return text[0:i + 1], i + 1
        else:
            return text[i - (k - 1):i + 1], k

    # 建立字典并给测试文件加字典标签
    print '添加开始:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for p in range(files_size // k):
        print '-' * 10, 'p =', p, '-' * 10

        # 建立字典
        src_files = [ner_result[i][:p * k] + ner_result[i][(p + 1) * k:] for i in range(4)]
        crf_dict, max_len = [[] for i in range(5)], 0
        for i in range(4):
            for src_file in src_files[i]:
                for line in open(ner_path + '/' + ner_folders[i] + '/' + src_file).readlines():
                    items = line[:-1].split('\t')
                    items[0] = unicode(items[0], 'utf-8')
                    max_len = max(max_len, len(items[0]))
                    if items[3] == "身体部位":
                        crf_dict[0].append(items[0])
                    elif items[3] == "症状和体征":
                        crf_dict[1].append(items[0])
                    elif items[3] == "疾病和诊断":
                        crf_dict[2].append(items[0])
                    elif items[3] == "检查和检验":
                        crf_dict[3].append(items[0])
                    else:
                        crf_dict[4].append(items[0])
        crf_dict = [list(set(d)) for d in crf_dict]

        # 为测试文件加字典标签
        orig_files, dest_files = [], []
        for i in range(4):
            orig_files.append(ori_result[i][p * k:(p + 1) * k])
            dest_files.append(crf_result[i][p * k:(p + 1) * k])
            for j in range(k):
                ori_txt = unicode(open(ori_path + '/' + ori_folders[i] + '/' + orig_files[i][j]).read())
                ori_txt = ori_txt.replace(' ', '').replace('\n', '').replace('\t', '')
                crf_txt = open(crf_path + '/' + crf_folders[i] + '/' + dest_files[i][j]).readlines()
                pos = len(ori_txt) - 1
                while pos >= 0:
                    tmp_words, length, = get_k_words(ori_txt, pos, max_len)
                    tmp_len = 0
                    for m in range(length):
                        flag = False
                        for n in range(5):
                            if tmp_words[m:length] in crf_dict[n]:
                                tmp_len = length - m
                                for ll in range(pos - tmp_len + 1, pos + 1):
                                    crf_txt[ll] = crf_txt[ll][:-1] + '\t' + names_env[n] + '\n'
                                flag = True
                                break
                            if m == length - 1 and n == 4:
                                tmp_len = length - m
                                crf_txt[pos] = crf_txt[pos][:-1] + '\t' + 'O' + '\n'
                        if flag: break
                    pos = pos - tmp_len
                f3 = open(crf_path + '/' + crf_folders[i] + '/' + dest_files[i][j], 'w')
                for line in crf_txt: f3.write(line)
                f3.close()
    print '添加结束:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__':
    # 文件夹路径
    path = os.getcwd()[:-5].replace('\\','/')           #总文件目录
    ori_path = unicode(path + '/original', 'utf8')       #源文件
    pos_path = unicode(path + '/pos_result', 'utf8')     #分词文件
    ner_path = unicode(path + '/manual_ner', 'utf8')     #标注文件
    crf_path = unicode(path + '/to_crfsuite', 'utf8')    #目标文件

    # 文件夹目录
    func = lambda name: int(name[:2])
    ori_folders = sorted(list(os.listdir(ori_path)), key = func)
    pos_folders = sorted(list(os.listdir(pos_path)), key = func)
    ner_folders = sorted(list(os.listdir(ner_path)), key = func)
    crf_folders = sorted(list(os.listdir(crf_path)), key = func)

    # print ori_folders, pos_folders, ner_folders, crf_folders

    # 将original文件、NER标准文件、pos文件一一匹配
    ori_func = lambda name: int(name[5:len(name)-16])
    pos_func = lambda name: int(name[5:len(name)-7])
    ner_func = lambda name: int(name[5:len(name)-4])
    ori_result = [sorted(list(os.listdir(ori_path + '/' + files)), key=ori_func) for files in ori_folders]
    pos_result = [sorted(list(os.listdir(pos_path + '/' + files)), key=pos_func) for files in pos_folders]
    ner_result = [sorted(list(os.listdir(ner_path + '/' + files)), key=ner_func) for files in ner_folders]
    all_files = [zip(ori, pos, ner) for ori, pos, ner in zip(ori_result, pos_result, ner_result)]

    # 实体类别名称 -- 中英文
    names_env = ['body', 'symp', 'dise', 'chec', 'cure']
    names_chs = ['身体部位', '症状和体征', '疾病和诊断', '检查和检验','治疗']

    # 文件预处理
    # one_to_crfsuite(all_files, '疾病和诊断')
    all_to_crfsuite(all_files)

    # 添加词典标记
    if dict_flag: add_dict(30)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
收集未标注的实体作为词典, 后处理未标注实体
"""

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

files_size = 300

# 找出应标注但未标注的词语加入词典
def search_unlabel(path, files_size):
    folders = list(os.listdir(path))
    func = lambda name: int(name[:-11])
    files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders]

    unlabel_dict = [[] for i in range(5)]
    map_dict = {'body':0, 'chec':1, 'dise':2, 'cure':3, 'symp':4}
    for i in range(4):
        for j in range(files_size):
            text = open( path + '/' + folders[i] + '/' + files[i][j]).readlines()
            k = 0
            while k < len(text):
                line = text[k][:-1].split('\t') 
                if line[2][0] == 'B' and line[3] == 'O':
                    entity, word = line[2][2:], ''
                    while text[k][:-1].split('\t')[2][2:] == entity:
                        word += text[k][:-1].split('\t')[0]
                        k += 1
                    unlabel_dict[map_dict[entity]].append(word + '\n')
                else:
                    k += 1
    return unlabel_dict

# 标注未标注词语
def tag_unlabel(ori_path, dest_path, unlabel_path):
	# 读取k个字,末字下标为i
    def get_k_words(text, i, k):
        if i + 1 < k:
            return text[0:i + 1], i + 1
        else:
            return text[i - (k - 1):i + 1], k

    # 读取unlabel_dict
    text = open(unlabel_path).readlines()
    unlabel_dict = [[] for i in range(5)]
    for i in range(5):
        k = 0
        while text[k] != '-'*20 + '\n':
            k += 1
        unlabel_dict[i] = [line[:-1] for line in text[:k]]
        text = text[k+1:]
    max_len = 24 # 暂时人为指定

    # 最大逆向匹配算法强行标注未标注实体
    for i in range(4):
        for j in range(files_size):
            print i, '-', j
            ori_folders = list(os.listdir(ori_path))
            dest_folders = list(os.listdir(dest_path))

            ori_func = lambda name: int(name[5:len(name) - 16])
            dest_func = lambda name: int(name[:-11])

            ori_files = [sorted(list(os.listdir(ori_path + '/' + files)), key=ori_func) for files in ori_folders]
            dest_files = [sorted(list(os.listdir(dest_path + '/' + files)), key=dest_func) for files in dest_folders]

            ori_txt = unicode(open(ori_path + '/' + ori_folders[i] + '/' + ori_files[i][j]).read())
            ori_txt = ori_txt.replace(' ', '').replace('\n', '').replace('\t', '')
            dest_txt = open(dest_path + '/' + dest_folders[i] + '/' + dest_files[i][j]).readlines()

            pos = len(ori_txt) - 1
            while pos >= 0:
                tmp_words, length, = get_k_words(ori_txt, pos, max_len)
                tmp_len = 0
                for m in range(length): # 当前截取内容
                    flag = False
                    for n in range(5):  # 5个实体类别
                        if tmp_words[m:length] in unlabel_dict[n]:
                            for k in range(pos - tmp_len + 1, pos + 1): # 检查是否均未标注
                                if dest_txt[k][:-1].split('\t')[-1] != 'O':
                                    break
                            else: 		# 后处理标注未标注的实体
                                tmp_len = length - m
                                if dest_txt[pos - tmp_len + 1][:-1].split('\t')[-1] == 'O':
                                    line = '\t'.join(dest_txt[pos - tmp_len + 1][:-1].split()[:-1])
                                    dest_txt[pos - tmp_len + 1] =  line + '\t' + 'B-' + names_env[n] + '\n'
                                    for ll in range(pos - tmp_len + 2, pos):
                                        line = '\t'.join(dest_txt[ll][:-1].split('\t')[:-1])
                                        dest_txt[ll] =  line + '\t' + 'I-' + names_env[n] + '\n'
                                    line = '\t'.join(dest_txt[pos][:-1].split('\t')[:-1])
                                    dest_txt[pos] = line + '\t' + 'E-' + names_env[n] + '\n'
                                flag = True
                                break
                        if m == length - 1 and n == 4: # 当前截取内容检测完毕
                            tmp_len = length - m
                    if flag: break
                pos = pos - tmp_len
            f3 = open(dest_path + '/' + dest_folders[i] + '/' + dest_files[i][j], 'w')
            f3.writelines(dest_txt)
            f3.close()

if __name__ == '__main__':
	names_env = ['body', 'symp', 'dise', 'chec', 'cure']
    path = unicode('C:/Users/zgdil/Desktop/CCKS项目/CRF_TEST/test_pred', 'utf-8')

    # 收集未标注实体作为词典
    unlabel_dict = search_unlabel(path, 300)
    w = open('unlabel_dict.txt','w')
    for i in range(5):
        w.writelines(set(unlabel_dict[i]))
        w.write('-'*20 + '\n')
    w.close()

    # 根据词典标注未标注实体
    ori_path = unicode('C:/Users/zgdil/Desktop/CCKS项目/CRF_TEST/original', 'utf-8')
    dest_path = unicode('C:/Users/zgdil/Desktop/unlabel', 'utf-8')
    tag_unlabel(ori_path, dest_path, 'unlabel_dict.txt', )
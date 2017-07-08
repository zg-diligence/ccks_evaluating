#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
半监督训练语料预处理文件,转换为crfsuite标准格式
NOTE:linux系统下运行
"""

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

files_size = 100

# 将分词结果转换为单字格式
def pos2word(un_str):
    tempdata = un_str.decode('utf8').split('#')
    result = tempdata[0][0] + '\t' + tempdata[1] + 'B\n'
    for i in range(1, len(tempdata[0])):
        result += tempdata[0][i] + '\t' + tempdata[1] + 'I\n'
    return result

# 将数据转换成CRFsuite标准格式
def all_to_crfsuite(files):
    crf_folders = list(os.listdir(crf_path))
    for folder in crf_folders:
        for f in list(os.listdir(crf_path + '/' + folder)):
            os.remove(crf_path + '/' + folder + '/' + f)
    for i in range(4):
        for j in range(files_size):
            path1 = crf_path + '/' + crf_folders[i] + '/' + files[i][j][5:8]
            w1 = open(pos_path + '/' + pos_folders[i] + '/' + files[i][j])
            w2 = open(path1 + '-result.txt','w')
            for line in w1:
                word_list = line.split(' ')
                for k in range(len(word_list)-1):
                    w2.write(pos2word(word_list[k]))
            w1.close()
            w2.close()

if __name__ == '__main__':
    path = os.getcwd()[:-5].replace('\\','/')
    pos_path = unicode(path + '/test_pos', 'utf8')       #分词文件
    crf_path = unicode(path + '/test_crfsuite', 'utf8')  #目标文件

    pos_func = lambda name: int(name[5:8])
    pos_folders = list(os.listdir(pos_path))
    pos_files = [sorted(list(os.listdir(pos_path + '/' + folder)), key=pos_func) for folder in pos_folders]

    all_to_crfsuite(pos_files)
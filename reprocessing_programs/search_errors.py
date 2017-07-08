#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
找出预测结果文件中所有错误,将其位置写入目标文件
"""

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

files_size = 300

# 获取文件目录
path = unicode('path', 'utf8')
folders = list(os.listdir(path))
func = lambda name:int(name[:-11])
files = [sorted(list(os.listdir(path + '/' + files)), key=func) for files in folders]

# 分析错误位置并写入文件
err_file = open('error.txt', 'w')
for i in range(4):
    for j in range(files_size):
        text = open(path + '/' + folders[i] + '/' + files[i][j]).readlines()
        k, char = 0, ''
        while k < len(text):
            line = text[k][:-1].split('\t')
            if line[2] != line[3]:
                err_file.write(str(i+1) + '-' + str(j+1) + '\tline-' + str(k+1) + '\n')
                char = line[2][1:]
                while k < len(text) and text[k][:-1].split('\t')[2][1:] == char:
                    k += 1
            else:
                k += 1
err_file.close()

# 选取错误数目多于10的文件
errors = open('error.txt').readlines()
err_file = open('error.txt', 'w')
start, num, pre, = 0, 0, ''
for k in range(len(errors)):
    if errors[k][:-1].split('\t')[0] == pre:
        num += 1
    else:
        if num > 10:
            err_file.writelines(errors[start:k])
            err_file.write('\n\n')
        num, start = 1, k
        pre = errors[k][:-1].split('\t')[0]
err_file.close()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在预测结果文件中找出满足要求的文件
"""

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

files_size = 2605

entity = 'cure'
path = 'C:/Users/zgdil/Desktop/CCKS项目/CRF_TEST/'
dest_path = unicode(path + 'semi_' + entity, 'utf8')

# 获取源文件夹所有文件
src_path = unicode(path + 'semi_crfsuite', 'utf8')
folders = list(os.listdir(src_path))
func = lambda name:int(name[:-11])
files = [sorted(list(os.listdir(src_path + '/' + folder)), key=func) for folder in folders]

# 删除目标文件夹原有的文件
for f in list(os.listdir(dest_path)):
    os.remove(dest_path + '/' + f)

# 将指定文件写入目标文件夹
tar_file = open('semi_files.txt', 'w')
for i in range(4):
    for j in range(files_size):
        if j+1 in [2547, 2529, 2308, 2318, 2399, 2259]: continue
        size = float(os.path.getsize(src_path + '/' + folders[i] + '/' + files[i][j])) / 1024
        if size > 12 or size < 0.1: continue
        text = open(src_path + '/' + folders[i] + '/' + files[i][j]).readlines()
        num = 0
        for line in text:
            if line[:-1].split('\t')[-1] == 'B-' + entity: num += 1
        if num >= 4:
            open(dest_path + '/' + str(i+1) + '-' + files[i][j], 'w').writelines(text)
            tar_file.write((str(i+1) + '-' + files[i][j] + '\n').encode('utf8'))
tar_file.close()
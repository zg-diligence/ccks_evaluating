# usr/python/bin
# -*- coding: utf-8 -*-

"""
使用ltp model进行分词和词性标注
"""

import os
from pyltp import Segmentor
from pyltp import Postagger

cws_model_path = 'cws.model'
pos_model_path = 'pos.model'
segmentor = Segmentor()
segmentor.load(cws_model_path)
postagger = Postagger()
postagger.load(pos_model_path)

path = unicode('C:/Users/zgdil/Desktop/ccks/SCIR/test_original','utf-8')
w_path = unicode('C:/Users/zgdil/Desktop/ccks/SCIR/test_pos','utf-8')
func = lambda name: int(name[:2])
folders = sorted(list(os.listdir(path)), key=func)
w_folders = sorted(list(os.listdir(w_path)), key=func)
func = lambda name: int(name[5:-16])
files = [sorted(list(os.listdir(path + '/' + folder)), key=func) for folder in folders]

for i in range(4):
    for j in range(100):
        read_path = path + '/' + folders[i] + '/' + files[i][j]
        write_path = w_path + '/' + w_folders[i] + '/' + str(j+1)+'.txt'
        text = open(read_path).read().strip().split('。')
        text = [sent for sent in text if sent]
        seg_text = [segmentor.segment(sent) for sent in text]
        pos_tags = [postagger.postag(sent) for sent in seg_text]
        results = [zip(seg_sent, pos_sent) for seg_sent, pos_sent in zip(seg_text, pos_tags)]
        results = [' '.join([word[0] + '#' + word[1] for word in sent]) for sent in results]
        open(write_path, 'w').write(' 。#wp '.join(results) + ' 。#wp')

segmentor.release()
postagger.release()


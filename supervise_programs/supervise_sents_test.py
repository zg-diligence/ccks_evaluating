#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
有监督训练测试脚本
"""

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import supervise_sents as ccks

template1 = (
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2)),
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2)))

template2 = (
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)),
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2)))

template3 = (
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)),
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-2, -1, 0), (-1, 0, 1), (0, 1, 2)))

template4 = (
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2), (-1, 0, 1)),
    ((-2,), (-1,), (0,), (1,), (2,), (-2, -1), (-1, 0), (0, 1), (1, 2)))

if __name__ == '__main__':
    path = os.getcwd()[:-5].replace('\\','/')
    src_path = path + '/to_crfsuite'
    dest_path = path + '/test_pred'

    # for num in range(10, 20):
    #     ccks.supervise_main(src_path, dest_path, 30, template4, num)
    ccks.supervise_main(src_path, dest_path, 30, template3, 17)
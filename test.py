# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:46
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : test.py.py
# @Software: PyCharm
from detectron2.utils.registry import Registry
ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
@ROI_HEADS_REGISTRY.register()
class test(object):
    def __init__(self):
        raise Exception
if __name__ == '__main__':
    a = test()


"""
Detect 配置文件，全局参数
author: 王建坤
date: 2018-12-3
"""
import numpy as np
import tensorflow as tf
from nets import alexnet, mobilenet_v1, vgg, inception_v4, resnet_v2, inception_resnet_v2
from Detect import mynet
import os

# 分类类别
CLASSES = 5
# 图像尺寸
IMG_SIZE = 224
# 图像通道数
CHANNEL = 2
# 是否为非标准图像尺寸
GLOBAL_POOL = True
# 是否使用GPU
USE_GPU = True
if not USE_GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 模型名称 'Mobile'
MODEL_NAME = 'My'
# 数据集路径
date_set_name = 'sia_cig_'  # b_cig_data_
images_path = '../data/' + date_set_name + str(IMG_SIZE) + '.npy'
labels_path = '../data/' + date_set_name + str(IMG_SIZE) + '.npy'




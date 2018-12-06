"""
Detect 配置文件，全局参数
author: 王建坤
date: 2018-12-3
"""
import numpy as np
import tensorflow as tf
from nets import alexnet, mobilenet_v1, vgg, inception_v4, resnet_v2, inception_resnet_v2
from Detect import mynet

CLASSES = 2
IMG_SIZE = 128
CHANNEL = 1
GLOBAL_POOL = False
MODEL_NAME = 'My'

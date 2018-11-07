"""
Detect 模型网络结构（自己搭建的）
author: 王建坤
date: 2018-8-10
"""
import tensorflow as tf
import numpy as np


# 卷积层
def convolution(input_tensor, conv_height, conv_width, conv_deep, x_stride, y_stride, name, padding='SAME'):
    with tf.variable_scope(name):
        channel = int(input_tensor.get_shape()[-1])
        weights = tf.get_variable("weights", shape=[conv_height, conv_width, channel, conv_deep],
                                  initializer=tf.truncated_normal_initializer(stddev=0.025))
        bias = tf.get_variable("bias", shape=[conv_deep], initializer=tf.constant_initializer(0.025))
        conv = tf.nn.conv2d(input_tensor, weights, strides=[1, x_stride, y_stride, 1], padding=padding)
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return relu


# 最大池化层
def max_pool(input_tensor, height, width, x_stride, y_stride, name, padding="SAME"):
    return tf.nn.max_pool(input_tensor, ksize=[1, height, width, 1], strides=[1, x_stride, y_stride, 1],
                          padding=padding, name=name)


# 局部响应归一化层
def LRN(input_tensor, R, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(input_tensor, depth_radius=R, bias=bias, alpha=alpha, beta=beta,
                                              name=name)


# dropout层
def dropout(input_tensor, prob, name):
    return tf.nn.dropout(input_tensor, prob, name=name)


# 全连接层
def full_connect(input_tensor, in_dimension, out_dimension, relu_flag, name):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [in_dimension, out_dimension],
                                  initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2 / in_dimension)))
        bias = tf.get_variable("bias", [out_dimension], initializer=tf.constant_initializer(0.05))
        fc = tf.matmul(input_tensor, weights) + bias
        if relu_flag:
            fc = tf.nn.relu(fc)
        return fc


def inference(input_tensor, train):
    conv1 = convolution(input_tensor, 11, 11, 96, 4, 4, "conv1", padding="VALID")
    pool1 = max_pool(conv1, 3, 3, 2, 2, "pool1", "VALID")
    lrn1 = LRN(pool1, 2, 2e-05, 0.75, "lrn1")
    conv2 = convolution(lrn1, 5, 5, 256, 1, 1, "conv2")
    pool2 = max_pool(conv2, 3, 3, 2, 2, "pool2", "VALID")
    lrn2 = LRN(pool2, 2, 2e-05, 0.75, "lrn2")
    conv3 = convolution(lrn2, 3, 3, 384, 1, 1, "conv3")
    conv4 = convolution(conv3, 3, 3, 384, 1, 1, "conv4")
    conv5 = convolution(conv4, 3, 3, 256, 1, 1, "conv5")
    pool5 = max_pool(conv5, 3, 3, 2, 2, "pool5", "VALID")
    fcin = tf.reshape(pool5, [-1, 256 * 6 * 6])
    fc1 = full_connect(fcin, 256 * 6 * 6, 4096, True, "fc6")
    if train:
        fc1 = dropout(fc1, 0.8, "drop6")
    fc2 = full_connect(fc1, 4096, 2, True, "fc7") # 2048
    # if train:
    #     fc2 = dropout(fc2, 1, "drop7")
    fc3 = full_connect(fc2, 2, 2, True, "fc8")
    return fc3

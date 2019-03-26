"""
Siamese 训练 -- 多分类网络结构
author: 王建坤
date: 2018-8-14
"""
import tensorflow as tf


def inference(inputs, keep_prob):
    """
        两层的神经网络进行多分类
    """
    with tf.name_scope('mul_fc1') as scope:
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[inputs.shape[1], 64], stddev=0.05, mean=0), name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(64), name='b_fc1')
        fc1 = tf.add(tf.matmul(inputs, w_fc1), b_fc1)
    with tf.name_scope('mul_relu_fc1') as scope:
        relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')
    with tf.name_scope('mul_drop_1') as scope:
        drop_1 = tf.nn.dropout(relu_fc1, keep_prob=keep_prob, name='drop_1')
    with tf.name_scope('mul_bn_fc1') as scope:
        bn_fc1 = tf.layers.batch_normalization(drop_1, name='bn_fc1')

    with tf.name_scope('mul_fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[64, 10], stddev=0.05, mean=0), name='w_fc1')
        b_fc2 = tf.Variable(tf.zeros(64), name='b_fc2')
        fc2 = tf.add(tf.matmul(bn_fc1, w_fc2), b_fc2)

    return fc2

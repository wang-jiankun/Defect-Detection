"""
Siamese 模型网络结构--基于LeNet5
author: 王建坤
date: 2018-8-14
"""
import tensorflow as tf


def inference(inputs, keep_prob):
    initer = tf.truncated_normal_initializer(stddev=0.1)

    with tf.name_scope('conv1') as scope:
        w1 = tf.get_variable('w1', dtype=tf.float32, shape=[11, 11, 3, 4], initializer=initer)
        b1 = tf.get_variable('b1', dtype=tf.float32, initializer=tf.constant(0.01, shape=[4], dtype=tf.float32))
        conv1 = tf.nn.conv2d(inputs, w1, strides=[1, 1, 1, 1], padding='VALID', name='conv1')
    with tf.name_scope('relu1') as scope:
        relu1 = tf.nn.relu(tf.add(conv1, b1), name='relu1')
    with tf.name_scope('max_pool1') as scope:
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool1')

    with tf.name_scope('conv2') as scope:
        w2 = tf.get_variable('w2', dtype=tf.float32, shape=[5, 5, 4, 8], initializer=initer)
        b2 = tf.get_variable('b2', dtype=tf.float32, initializer=tf.constant(0.01, shape=[8], dtype=tf.float32))
        conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
    with tf.name_scope('relu2') as scope:
        relu2 = tf.nn.relu(conv2 + b2, name='relu2')
    with tf.name_scope('max_pool2') as scope:
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool2')

    with tf.name_scope('conv3') as scope:
        w3 = tf.get_variable('w3', dtype=tf.float32, shape=[3, 3, 8, 8], initializer=initer)
        b3 = tf.get_variable('b3', dtype=tf.float32, initializer=tf.constant(0.01, shape=[8], dtype=tf.float32))
        conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='VALID', name='conv3')
    with tf.name_scope('relu3') as scope:
        relu3 = tf.nn.relu(conv3 + b3, name='relu3')
    with tf.name_scope('max_pool3') as scope:
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool3')

    dim = pool3.get_shape()[1]*pool3.get_shape()[2]*pool3.get_shape()[3]
    # print(pool3.get_shape()[1], pool3.get_shape()[2], pool3.get_shape()[3])

    with tf.name_scope('fc1') as scope:
        x_flat = tf.reshape(pool3, shape=[-1, dim])
        w_fc1 = tf.get_variable('w_fc1', dtype=tf.float32, shape=[int(dim), 128], initializer=initer)
        b_fc1 = tf.get_variable('b_fc1', dtype=tf.float32, initializer=tf.constant(0.01, shape=[128], dtype=tf.float32))
        fc1 = tf.add(tf.matmul(x_flat, w_fc1), b_fc1)
    with tf.name_scope('relu_fc1') as scope:
        relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')
    # with tf.name_scope('bn_fc1') as scope:
    #     bn_fc1 = tf.layers.batch_normalization(relu_fc1, name='bn_fc1')
    with tf.name_scope('drop_1') as scope:
        drop_1 = tf.nn.dropout(relu_fc1, keep_prob=keep_prob, name='drop_1')

    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.get_variable('w_fc2', dtype=tf.float32, shape=[128, 32], initializer=initer)
        b_fc2 = tf.get_variable('b_fc2', dtype=tf.float32, initializer=tf.constant(0.01, shape=[32], dtype=tf.float32))
        fc2 = tf.add(tf.matmul(drop_1, w_fc2), b_fc2)

    return fc2


# 损失函数1
# y=1 表示同一类
def siamese_loss(out1, out2, y, margin=5.0):
    Q = tf.constant(margin, name="Q", dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2), 1))
    # 同类
    pos = tf.multiply(tf.multiply(y, 2/Q), tf.square(E_w))
    # 不同类
    neg = tf.multiply(tf.multiply(1-y, 2*Q), tf.exp(-2.77/Q*E_w))
    loss = pos + neg
    loss = tf.reduce_mean(loss)
    return loss


# 损失函数2
def loss_spring(out1, out2, y, margin=5.0):
    eucd2 = tf.reduce_sum(tf.square(out1-out2), 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    # (1-yi)*||CNN(p1i)-CNN(p2i)||^2 + yi*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
    # 同类
    pos = tf.multiply(y, eucd, name="pos_loss")
    # 不同类
    # neg = tf.multiply(1 - y, tf.pow(tf.maximum(C - eucd, 0.0), 2), name="neg_loss")
    neg = tf.multiply(1-y, tf.maximum(C-eucd, 0.0), name="neg_loss")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss


# def inference(inputs, keep_prob):
#     with tf.name_scope('conv1') as scope:
#         w1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], stddev=0.05), name='w1')
#         b1 = tf.Variable(tf.zeros(32), name='b1')
#         conv1 = tf.nn.conv2d(inputs, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
#     with tf.name_scope('relu1') as scope:
#         relu1 = tf.nn.relu(tf.add(conv1, b1), name='relu1')
#
#     with tf.name_scope('conv2') as scope:
#         w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.05), name='w2')
#         b2 = tf.Variable(tf.zeros(64), name='b2')
#         conv2 = tf.nn.conv2d(relu1, w2, strides=[1, 2, 2, 1], padding='SAME', name='conv2')
#     with tf.name_scope('relu2') as scope:
#         relu2 = tf.nn.relu(conv2 + b2, name='relu2')
#
#     with tf.name_scope('conv3') as scope:
#         w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.05), name='w3')
#         b3 = tf.Variable(tf.zeros(128), name='b3')
#         conv3 = tf.nn.conv2d(relu2, w3, strides=[1, 2, 2, 1], padding='SAME')
#     with tf.name_scope('relu3') as scope:
#         relu3 = tf.nn.relu(conv3 + b3, name='relu3')
#
#     with tf.name_scope('fc1') as scope:
#         x_flat = tf.reshape(relu3, shape=[-1, 7 * 7 * 128])
#         w_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 128, 1024], stddev=0.05, mean=0), name='w_fc1')
#         b_fc1 = tf.Variable(tf.zeros(1024), name='b_fc1')
#         fc1 = tf.add(tf.matmul(x_flat, w_fc1), b_fc1)
#     with tf.name_scope('relu_fc1') as scope:
#         relu_fc1 = tf.nn.relu(fc1, name='relu_fc1')
#     with tf.name_scope('bn_fc1') as scope:
#         bn_fc1 = tf.layers.batch_normalization(relu_fc1, name='bn_fc1')
#     with tf.name_scope('drop_1') as scope:
#         drop_1 = tf.nn.dropout(bn_fc1, keep_prob=keep_prob, name='drop_1')
#
#     with tf.name_scope('fc2') as scope:
#         w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.05, mean=0), name='w_fc2')
#         b_fc2 = tf.Variable(tf.zeros(512), name='b_fc2')
#         fc2 = tf.add(tf.matmul(drop_1, w_fc2), b_fc2)
#     with tf.name_scope('relu_fc2') as scope:
#         relu_fc2 = tf.nn.relu(fc2, name='relu_fc2')
#     with tf.name_scope('bn_fc2') as scope:
#         bn_fc2 = tf.layers.batch_normalization(relu_fc2, name='bn_fc2')
#     with tf.name_scope('drop_2') as scope:
#         drop_2 = tf.nn.dropout(bn_fc2, keep_prob=keep_prob, name='drop_2')
#
#     with tf.name_scope('fc3') as scope:
#         w_fc3 = tf.Variable(tf.truncated_normal(shape=[512, 64], stddev=0.05, mean=0), name='w_fc3')
#         b_fc3 = tf.Variable(tf.zeros(64), name='b_fc3')
#         fc3 = tf.add(tf.matmul(drop_2, w_fc3), b_fc3)
#
#     return fc3

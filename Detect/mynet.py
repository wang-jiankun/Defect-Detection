"""
缺陷检测，自定义网络结构
author: 王建坤
date: 2018-12-5
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time

IMG_SIZE = 128


def mynet_v1(inputs, num_classes=10, is_training=True, scope=None):
    with tf.variable_scope(scope, 'Mynet_v1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = slim.conv2d(inputs, 16, [11, 11], stride=4, padding='VALID', scope='Conv_1')
                # net = slim.batch_norm(net, scope='Conv_1_bn')
                net = slim.conv2d(net, 32, [1, 1], stride=1, scope='Conv_2')
                net = slim.separable_conv2d(net, None, [5, 5], depth_multiplier=1.0, stride=3, scope='Conv_3_dw')
                net = slim.conv2d(net, 64, [1, 1], stride=1, scope='Conv_3_pw')
                net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1.0, stride=2, scope='Conv_4_dw')
                net = slim.conv2d(net, 64, [1, 1], stride=1, scope='Conv_4_pw')
                net = slim.conv2d(net, 64, [5, 5], padding='VALID', scope='Conv_5')
                pre = slim.conv2d(net, num_classes, [1, 1], scope='Conv_6')
                pre = tf.squeeze(pre, [1, 2], name='squeezed')
                # pre = slim.softmax(pre, scope='Predictions')

            return pre


def train():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1], name="x_input")
    y_ = tf.placeholder(tf.uint8, [None, 3], name="y_input")

    images = np.random.randint(255, size=(100, IMG_SIZE, IMG_SIZE, 1))
    labels = np.random.randint(3, size=(100, 3))

    y = mynet_v1(x, num_classes=3, scope=None)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_, name='entropy')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 0

        while step < 10:
            sess.run(train_op, feed_dict={x: images, y_: labels})
            step += 1
            print(step)


def predict():
    x = tf.placeholder(tf.float32, [None, 2000, 1000, 1], name="x_input")
    y = mynet_v1(x, num_classes=3, scope=None)

    image = np.random.randint(255, size=(1, 2000, 1000, 1))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(y, feed_dict={x: image})

        start_time = time.clock()
        res = sess.run(y, feed_dict={x: image})
        end_time = time.clock()
        print('run time: ', end_time - start_time)
        print(res.shape)


if __name__ == '__main__':
    images = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1], name="x_input")
    res = mynet_v1(images, 2)
    print(res)
    # train()
    # predict()



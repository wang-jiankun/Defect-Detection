"""
手机图像分割，学习目标矩形框
author: 王建坤
date: 2018-10-25
"""
import tensorflow as tf
import numpy as np
import os
import time
import tensorflow.contrib.slim as slim
from sklearn.model_selection import train_test_split
from nets import alexnet
from Detect import utils
from PIL import Image
import cv2


def lenet5(inputs):
    inputs = tf.reshape(inputs, [-1, 300, 300, 3])
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='conv1')
    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2')
    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 500, scope='fully4')
    net = slim.fully_connected(net, 10, scope='fully5')
    return net


MAX_STEP = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.96
IMG_SIZE = 224
SHOW_SIZE = 800
CLASSES = 8
BATCH_SIZE = 32

INFO_STEP = 20
SAVE_STEP = 200
GLOBAL_POOL = False


def train(inherit=False, model='Alex'):
    # 加载数据集
    images = np.load('../data/card_data.npy')
    labels = np.load('../data/card_label.npy')
    train_data, val_data, train_label, val_label = train_test_split(images, labels, test_size=0.2, random_state=222)

    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, CLASSES], name='y-input')
    my_global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    # 前向传播
    if model == 'Alex':
        log_path = "../log/Alex"
        model_name = 'alex.ckpt'
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=True,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)  # 输入不是规定的尺寸时，需要global_pool
    else:
        log_path = '../log/My'
        model_name = 'my.ckpt'
        y, _ = lenet5(x)

    # 交叉熵、损失值、优化器、准确率
    loss = tf.reduce_mean(tf.sqrt(tf.square(y_ - y)+0.0000001))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, my_global_step, 100, LEARNING_RATE_DECAY)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=my_global_step)

    # 模型保存器、初始化
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    tf.summary.scalar("loss", loss)
    # tf.summary.histogram("W1", W)
    merged_summary_op = tf.summary.merge_all()

    # 训练迭代
    with tf.Session() as sess:
        summary_writer1 = tf.summary.FileWriter('../log/curve/train', sess.graph)
        summary_writer2 = tf.summary.FileWriter('../log/curve/test')
        step = 0
        if inherit:
            ckpt = tf.train.get_checkpoint_state(log_path)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(model, 'continue train from %s:' % global_step)
                step = int(global_step)
            else:
                print('Error: no checkpoint file found')
                return
        else:
            print(model, 'restart train:')
            sess.run(init)

        # 迭代
        while step < MAX_STEP:
            start_time = time.clock()
            image_batch, label_batch = utils.get_batch(train_data, train_label, BATCH_SIZE)

            # 训练，损失值和准确率
            _, train_loss = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
            end_time = time.clock()
            runtime = end_time - start_time

            step += 1
            # 训练信息、曲线和保存模型
            if step % INFO_STEP == 0 or step == MAX_STEP:
                summary_str = sess.run(merged_summary_op, feed_dict={x: image_batch, y_: label_batch})
                summary_writer1.add_summary(summary_str, step)
                test_loss, summary_str = sess.run([loss, merged_summary_op], feed_dict={x: val_data, y_: val_label})
                summary_writer2.add_summary(summary_str, step)
                print('step: %d, runtime: %.2f, train loss: %.4f, test loss: %.4f' %
                      (step, runtime, train_loss, test_loss))

            if step % SAVE_STEP == 0:
                checkpoint_path = os.path.join(log_path, model_name)
                saver.save(sess, checkpoint_path, global_step=step)


def predict(root_path, model='Alex'):
    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])

    # 模型保存路径，前向传播
    if model == 'Alex':
        log_path = "../log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=True,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)        # 输入不是规定的尺寸时，需要global_pool
    else:
        log_path = '../log/My'
        y, _ = lenet5(x)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 恢复模型权重
        print('Reading checkpoints: ', model)
        ckpt = tf.train.get_checkpoint_state(log_path)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('Error: no checkpoint file found')
            return

        file_list = os.listdir(root_path)
        for file_name in file_list:
            if file_name.split('.')[-1] == 'jpg':
                img = Image.open(os.path.join(root_path, file_name))
                img_show = img.resize((SHOW_SIZE, SHOW_SIZE))
                img_show = np.array(img_show, np.uint8)
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img, np.uint8)
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=3)
                pre = np.squeeze(sess.run(y, feed_dict={x: img}))
                print('predict:', pre)

                points = [[pre[0] * SHOW_SIZE, pre[1] * SHOW_SIZE], [pre[2] * SHOW_SIZE, pre[3] * SHOW_SIZE],
                          [pre[4] * SHOW_SIZE, pre[5] * SHOW_SIZE], [pre[6] * SHOW_SIZE, pre[7] * SHOW_SIZE]]
                points = np.int0(points)
                cv2.polylines(img_show, [points], True, (0, 0, 255), 1)
                cv2.imshow('img_show', img_show)
                cv2.waitKey()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    train()
    # predict('../data/card')

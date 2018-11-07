"""
Siamese 训练 -- 相似度损失函数
author: 王建坤
date: 2018-9-30
"""
import os
import numpy as np
import tensorflow as tf
from Siamese import inference, utils

MAX_STEP = 2000
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.98
IMG_SIZE = 224
# 训练信息和保存权重的gap
INFO_STEP = 20
SAVE_STEP = 200


def train(inherit=False):
    # 加载数据集
    images = np.load('E:/dataset/npy/train_data.npy')
    labels = np.load('E:/dataset/npy/train_label.npy')
    total_train = images.shape[0]
    images, labels = utils.shuffle_data(images, labels)

    # 占位符
    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    with tf.variable_scope('y') as scope:
        y = tf.placeholder(tf.float32, shape=[None])

    with tf.name_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    # 前向传播
    with tf.variable_scope('siamese') as scope:
        out1 = inference.inference(x1, keep_prob)
        # 参数共享，不会生成两套参数。注意定义variable时要使用get_variable()
        scope.reuse_variables()
        out2 = inference.inference(x2, keep_prob)

    # 损失函数和优化器
    with tf.variable_scope('metrics') as scope:
        loss = inference.loss_spring(out1, out2, y)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, my_global_step, 100, LEARNING_RATE_DECAY)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=my_global_step)

    saver = tf.train.Saver(tf.global_variables())

    # 模型保存路径
    log_dir = "E:/alum/log/Siamese"

    with tf.Session() as sess:
        step = 0
        if inherit:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Siamese continue train from %s' % global_step)
                step = int(global_step)
            else:
                print('No checkpoint file found')
                return
        else:
            print('restart train')
            sess.run(tf.global_variables_initializer())

        while step < MAX_STEP:
            # 获取一对batch的数据集
            xs_1, ys_1 = utils.get_batch(images, labels, BATCH_SIZE, total_train)
            xs_2, ys_2 = utils.get_batch(images, labels, BATCH_SIZE, total_train)
            # 判断对应的两个标签是否相等
            y_s = np.array(ys_1 == ys_2, dtype=np.float32)

            _, y1, y2, train_loss = sess.run([train_op, out1, out2, loss],
                                             feed_dict={x1: xs_1, x2: xs_2, y: y_s, keep_prob: 0.6})

            # 训练信息和保存模型
            step += 1
            if step % INFO_STEP == 0 or step == MAX_STEP:
                print('step: %d, loss: %.4f' % (step, train_loss))

            if step % SAVE_STEP == 0 or step == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'Siamese_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()

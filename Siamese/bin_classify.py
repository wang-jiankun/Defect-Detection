"""
Siamese 训练 -- 二分类
author: 王建坤
date: 2018-10-15
"""
import tensorflow as tf
import numpy as np
import os
from Siamese import inference, utils

MAX_STEP = 600
LEARNING_RATE = 0.01
# 训练信息和保存权重的gap
INFO_STEP = 20
SAVE_STEP = 200

BATCH_SIZE = 128
IMG_SIZE = 299


def train(inherit=False):
    """
    用二分类来训练 Siamese
    """
    # 加载数据集
    images = np.load('../data/data_'+str(IMG_SIZE)+'.npy')
    labels = np.load('../data/label_'+str(IMG_SIZE)+'.npy')
    images, labels = utils.shuffle_data(images, labels)

    # 占位符
    with tf.variable_scope('bin_input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    with tf.variable_scope('bin_input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    with tf.variable_scope('bin_y') as scope:
        y = tf.placeholder(tf.float32, shape=[None, 1])
    with tf.name_scope('bin_keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)

    # 前向传播
    with tf.variable_scope('bin_siamese') as scope:
        out1 = inference.inference(x1, keep_prob)
        # 参数共享，不会生成两套参数
        scope.reuse_variables()
        out2 = inference.inference(x2, keep_prob)

    # 增加二分类层
    with tf.name_scope('bin_c') as scope:
        w_bc = tf.Variable(tf.truncated_normal(shape=[64, 1], stddev=0.05, mean=0), name='w_bc')
        b_bc = tf.Variable(tf.zeros(1), name='b_bc')
        out12 = tf.concat((out1, out2), 1, name='out12')
        pre = tf.add(tf.matmul(out12, w_bc), b_bc)

    # 损失函数和优化器
    with tf.variable_scope('metrics') as scope:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pre, name='entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())

    # 模型保存路径
    log_path = '../log/Siamese'

    with tf.Session() as sess:
        step = 0
        if inherit:
            ckpt = tf.train.get_checkpoint_state(log_path)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Siamese continue train from %s:' % global_step)
                step = int(global_step)
            else:
                print('Error: no checkpoint file found')
                return
        else:
            print('Siamese restart train:')
            sess.run(init)

        while step < MAX_STEP:
            # 获取一对batch的数据集
            x_1, y_1 = utils.get_batch(images, labels, BATCH_SIZE)
            x_2, y_2 = utils.get_batch(images, labels, BATCH_SIZE)
            # 判断对应的两个标签是否相等
            y_s = np.array(y_1 == y_2, dtype=np.uint8)
            y_s = np.expand_dims(y_s, axis=1)

            _, train_loss = sess.run([train_op, loss], feed_dict={x1: x_1, x2: x_2, y: y_s, keep_prob: 0.8})

            step += 1
            # 训练信息和保存模型
            if step % INFO_STEP == 0 or step == MAX_STEP:
                print('step: %d, loss: %.4f' % (step, train_loss))

            if step % SAVE_STEP == 0:
                checkpoint_path = os.path.join(log_path, 'sia_bin.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()

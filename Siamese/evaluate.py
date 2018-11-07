"""
Siamese 评估
author: 王建坤
date: 2018-8-16
"""
import tensorflow as tf
import numpy as np
from Siamese import inference, utils

BATCH_SIZE = 132
CLASSES = 3
CODE_LEN = 32


def evaluate():
    # 加载数据集
    images = np.load('E:/dataset/npy/train_data.npy')
    labels = np.load('E:/dataset/npy/train_label.npy')
    total_test = images.shape[0]

    # 占位符
    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])
    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])
    with tf.variable_scope('y') as scope:
        y = tf.placeholder(tf.float32, shape=[None])

    with tf.name_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)

    # 前向传播
    with tf.variable_scope('siamese') as scope:
        out1 = inference.inference(x1, keep_prob)
        # 参数共享，不会生成两套参数
        scope.reuse_variables()
        out2 = inference.inference(x2, keep_prob)

    # 损失函数和优化器
    with tf.variable_scope('metrics') as scope:
        loss = inference.loss_spring(out1, out2, y)

    saver = tf.train.Saver(tf.global_variables())

    # 模型保存路径
    log_dir = "E:/alum/log/Siamese"

    with tf.Session() as sess:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return

        # 各个类别的样本数和平均编码
        sample_num = np.zeros(CLASSES)
        code_sum = np.zeros((CLASSES, CODE_LEN))
        for i in range(total_test):
            xs_1 = images[i:i+1]
            xs_1 = np.expand_dims(xs_1, axis=3)
            l_1 = np.argmax(labels[i])
            y1 = sess.run(out1, feed_dict={x1: xs_1, keep_prob: 1})
            sample_num[l_1] += 1
            code_sum[l_1] += np.squeeze(y1)
        code_mean = code_sum/np.expand_dims(sample_num, axis=1)
        # print('sample_num: ', '\n', sample_num, '\n', 'code_mean: ',  code_mean)

        # 类别间的距离
        class_diff = np.zeros((CLASSES, CLASSES))
        for i in range(CLASSES):
            for j in range(i, CLASSES):
                class_diff[i][j] = class_diff[j][i] = np.mean(np.square(code_mean[i]-code_mean[j]))
        print('code diff between class: ', '\n', class_diff)


if __name__ == '__main__':
    evaluate()

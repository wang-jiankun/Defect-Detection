"""
Siamese 训练 -- 多分类训练
author: 王建坤
date: 2018-8-14
"""
import tensorflow as tf
import numpy as np
import os
from Siamese_multi import utils
from Siamese import inference_2

learning_rate = 0.01
iterations = 100
batch_size = 64


def multi_classify_train():
    """
        用Siamese提取的特征进行多分类
    """
    # 加载数据集
    feature_vectors = np.load('F:/DefectDetection/npy_data/feature_vectors.npy')
    labels = np.load('F:/DefectDetection/npy_data/train_label.npy')

    total_train = feature_vectors.shape[0]

    # 占位符
    input_vectors = tf.placeholder(tf.float32, shape=[None, 64], name='input_vectors')
    y = tf.placeholder(tf.float32, shape=[None, 3])
    keep_prob = tf.placeholder(tf.float32, name='multi_keep_prob')

    # 前向传播
    result = inference_2.inference(input_vectors, keep_prob)

    # 损失函数和优化器
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=result, name='multi_entropy')
    loss = tf.reduce_mean(cross_entropy, name='multi_loss')
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # 模型保存路径
    log_dir = "F:/DefectDetection/log/Siamese_multi"
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start = 0
        for iter in range(iterations):
            # 获取一个batch的数据集
            vectors_batch = utils.get_batch(feature_vectors, start, batch_size, total_train)
            label_batch = utils.get_batch(labels, start, batch_size, total_train)
            _, train_loss = sess.run([train_op, loss], feed_dict={input_vectors: vectors_batch,
                                                                   y: label_batch, keep_prob: 0.6})
            start += batch_size
            # if iter % 100 == 1:
            #     print('iter {},train loss {}'.format(iter, train_loss))

            # 训练信息和保存模型
            if (iter + 11) % 10 == 0 or (iter + 1) == iterations:
                print('iter: %d, loss: %.4f' % (iter, train_loss))
                checkpoint_path = os.path.join(log_dir, 'Siamese_multi_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=iter)


if __name__ == '__main__':
    multi_classify_train()

"""
Siamese 缓存数据 -- 缓存用Siamese提取每个样本的特征向量
author: 王建坤
date: 2018-8-14
"""
import tensorflow as tf
import numpy as np
from Siamese import inference


def cache_data():
    """
        缓存用Siamese提取每个样本的特征向量
    """
    # 加载数据集
    images = np.load('F:/DefectDetection/npy_data/train_data.npy')
    # 根据自己的图像尺寸修改shape
    images_input = tf.reshape(images, [-1, 28, 28, 1])

    # 前向传播
    result = inference.inference(images_input, 1.0)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'F:/DefectDetection/log/Siamese')
        feature_vectors = sess.run(result)

    # 保存为npy文件
    np.save('F:/DefectDetection/npy_data/feature_vectors.npy', feature_vectors)


if __name__ == '__main__':
    cache_data()

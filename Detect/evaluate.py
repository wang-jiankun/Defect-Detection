"""
Detect 性能评估
author: 王建坤
date: 2018-9-10
"""
import math
import numpy as np
import tensorflow as tf
from nets import alexnet, vgg, inception_v4, resnet_v2
from sklearn.model_selection import train_test_split


BATCH_SIZE = 64
IMG_SIZE = 299
CLASSES = 8
GLOBAL_POOL = True


def evaluate(model='Alex'):
    """
    评估模型
    :param model: model name
    :return: none
    """
    # 预测为某类的样本个数，某类预测正确的样本个数
    pre_pos = np.zeros(CLASSES, dtype=np.uint16)
    true_pos = np.zeros(CLASSES, dtype=np.uint16)

    # 加载测试集
    images = np.load('../data/data_299.npy')
    labels = np.load('../data/label_299.npy')
    _, val_data, _, val_label = train_test_split(images, labels, test_size=0.2, random_state=222)
    test_num = val_data.shape[0]
    # 存放预测结果
    res = np.zeros(test_num, dtype=np.uint16)

    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name="x_input")
    y_ = tf.placeholder(tf.uint8, [None], name="y_input")

    # 模型保存路径，模型名，预训练文件路径，前向传播
    if model == 'Alex':
        log_dir = "../log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=True,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)  # 输入不是规定的尺寸时，需要global_pool
    elif model == 'VGG':
        log_dir = "../log/VGG"
        y, _ = vgg.vgg_16(x,
                          num_classes=CLASSES,
                          is_training=True,
                          dropout_keep_prob=1.0,
                          spatial_squeeze=True,
                          global_pool=GLOBAL_POOL)
    elif model == 'Incep4':
        log_dir = "E:/alum/log/Incep4"
        y, _ = inception_v4.inception_v4(x, num_classes=CLASSES,
                                         is_training=True,
                                         dropout_keep_prob=1.0,
                                         reuse=None,
                                         scope='InceptionV4',
                                         create_aux_logits=True)
    elif model == 'Res':
        log_dir = "E:/alum/log/Res"
        y, _ = resnet_v2.resnet_v2_50(x,
                                      num_classes=CLASSES,
                                      is_training=True,
                                      global_pool=GLOBAL_POOL,
                                      output_stride=None,
                                      spatial_squeeze=True,
                                      reuse=None,
                                      scope='resnet_v2_50')
    else:
        print('Error: model name not exist')
        return

    # 预测结果
    y_pre = tf.argmax(y, 1)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # 恢复模型权重
        print('Reading checkpoints of', model)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, step is %s' % global_step)
        else:
            print('Error: no checkpoint file found')
            return

        # 遍历一次测试集需要次数
        num_iter = int(math.ceil(test_num / test_num))

        step = 0
        start = 0
        while step < num_iter:
            # 获取一个 batch
            if step == num_iter-1:
                end = test_num
            else:
                end = start+BATCH_SIZE
            image_batch = val_data[start:end]
            label_batch = val_label[start:end]
            # 准确率和预测结果统计信息
            pre = sess.run(y_pre, feed_dict={x: image_batch, y_: label_batch})
            pre = np.squeeze(pre)
            res[start:end] = pre
            start += BATCH_SIZE
            step += 1
    # 计算准确率
    normal_num = 0
    for i in range(test_num):
        pre_pos[res[i]] += 1
        if res[i] == val_label[i]:
            true_pos[res[i]] += 1
        if val_label[i] == 0:
            normal_num += 1

    precision = true_pos/pre_pos
    print('测试样本数：', test_num)
    print('预测数：', pre_pos)
    print('正确数：', true_pos)
    print('各类精度：', precision)
    print('准确率：', np.sum(true_pos)/test_num)
    print('平均准确率：', np.mean(precision))
    print('缺陷漏检率：', (pre_pos[0]-true_pos[0])/test_num)
    print('正常漏检率：', (normal_num-true_pos[0])/test_num)


if __name__ == '__main__':
    evaluate()

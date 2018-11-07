"""
模型融合--根据预测结果进行融合
author: 王建坤
date: 2018-9-29
"""
import math
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from nets import alexnet, vgg, inception_v4, resnet_v2


BATCH_SIZE = 32
IMG_SIZE1 = 224
IMG_SIZE2 = 299
CLASSES = 12
GLOBAL_POOL = True
test_dir = 'E:/dataset/alum/guangdong_round1_test_a_20180916'
word_class = {'norm': 0, 'defect1': 1, 'defect2': 2, 'defect3': 3, 'defect4': 4, 'defect5': 5, 'defect6': 6,
              'defect7': 7, 'defect8': 8, 'defect9': 9, 'defect10': 10, 'defect11': 11}
class_label = ['norm', 'defect1', 'defect2', 'defect3', 'defect4', 'defect5',
               'defect6', 'defect7', 'defect8', 'defect9', 'defect10', 'defect11']


def choose_model(x, model):
    """
    选择模型
    :param x:
    :param model:
    :return:
    """
    # 模型保存路径，模型名，预训练文件路径，前向传播
    if model == 'Alex':
        log_dir = "E:/alum/log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=True,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)  # 输入不是规定的尺寸时，需要global_pool
    elif model == 'VGG':
        log_dir = "E:/alum/log/VGG"
        y, _ = vgg.vgg_16(x,
                          num_classes=CLASSES,
                          is_training=True,
                          dropout_keep_prob=1.0,
                          spatial_squeeze=True,
                          global_pool=GLOBAL_POOL)
    elif model == 'VGG2':
        log_dir = "E:/alum/log/VGG2"
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

    return y, log_dir


def evaluate_model_merge(model1='VGG2', model2='VGG', submit=True):
    """
    模型融合，并评估结果
    :param model1:
    :param model2:
    :param submit:
    :return:
    """
    # 存放模型1、2的预测结果
    image_name_list = os.listdir(test_dir)
    image_labels = pd.read_csv('E:/WJK_File/Python_File/Defect_Detection/Detect/test_label.csv')
    res, res1, res2, res_name, labels = [], [], [], [], []

    # model 1
    g1 = tf.Graph()
    with g1.as_default():
        x1 = tf.placeholder(tf.float32, [None, IMG_SIZE1, IMG_SIZE1, 3], name="x_input_1")
        y1, log_dir1 = choose_model(x1, model1)
        y1 = tf.nn.softmax(y1)
        saver1 = tf.train.Saver()
    sess1 = tf.Session(graph=g1)
    print("Reading checkpoints of model1")
    ckpt = tf.train.get_checkpoint_state(log_dir1)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver1.restore(sess1, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('Error: no checkpoint file found')
        return

    # model 1
    g2 = tf.Graph()
    with g2.as_default():
        x2 = tf.placeholder(tf.float32, [None, IMG_SIZE2, IMG_SIZE2, 3], name="x_input_2")
        y2, log_dir2 = choose_model(x2, model2)
        y2 = tf.nn.softmax(y2)
        saver2 = tf.train.Saver()
    sess2 = tf.Session(graph=g2)
    print("Reading checkpoints of model2")
    ckpt = tf.train.get_checkpoint_state(log_dir2)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver2.restore(sess2, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('model2 checkpoint file is not found ')
        return

    for image_name in image_name_list:
        label = image_labels[image_labels.iloc[:, 0].isin([image_name])].iloc[0, 1]
        labels.append(word_class[label])
        image_path = os.path.join(test_dir, image_name)
        img = Image.open(image_path)
        img1 = img.resize((IMG_SIZE1, IMG_SIZE1))
        img1 = np.array(img1, np.float32)
        img1 = np.expand_dims(img1, axis=0)
        pre1 = sess1.run(y1, feed_dict={x1: img1})
        max_index1 = np.argmax(pre1)
        max_value1 = np.max(pre1)
        res1.append([max_index1, max_value1])

        img2 = img.resize((IMG_SIZE2, IMG_SIZE2))
        img2 = np.array(img2, np.float32)
        img2 = np.expand_dims(img2, axis=0)
        pre2 = sess2.run(y2, feed_dict={x2: img2})
        max_index2 = np.argmax(pre2)
        max_value2 = np.max(pre2)
        res2.append([max_index2, max_value2])

    sess1.close()
    sess2.close()
    # print(res1, '\n', res2)
    img_num = len(res1)
    for i in range(img_num):
        if res1[i][0] == res2[i][0]:
            res.append(res1[i][0])
        else:
            if res1[i][1] > res2[i][1]:
                res.append(res1[i][0])
            else:
                res.append(res2[i][0])
            print(res1[i], res2[i], res[i], labels[i])
    print(res, '\n', labels)

    # 评估结果
    pre_pos = np.zeros(12)
    true_pos = np.zeros(12)
    for i in range(img_num):
        pre_pos[res[i]] += 1
        if res[i] == labels[i]:
            true_pos[res[i]] += 1
    precision = true_pos/pre_pos
    print(pre_pos, '\n', true_pos, '\n', precision)
    print('准确率：', np.sum(true_pos)/img_num, '\n', '平均准确率：', np.mean(precision))

    # 是否要保存结果为csv
    if submit:
        for i in res:
            res_name.append(class_label[i])
        print('image: ', image_name_list, '\n', 'class:', res_name)
        # 存为csv文件。两列：图片名，预测结果
        data = pd.DataFrame({'0': image_name_list, '1': res_name})
        data.to_csv("merge.csv", index=False, header=False)


if __name__ == '__main__':
    evaluate_model_merge()

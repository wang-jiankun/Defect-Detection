"""
Detect 铝型材表面缺陷检测--提交结果
author: 王建坤
date: 2018-9-17
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from PIL import Image, ImageFile
from nets import alexnet, vgg, inception_v4, resnet_v2
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = 12
IMG_SIZE = 299
GLOBAL_POOL = True


def submit(test_dir, model='VGG'):
    """
    测试集预测结果保存为 csv 文件
    :param test_dir: 测试集根路径
    :param model: 模型名称
    :return: none
    """
    # 类别名称
    print('running submit')
    tf.reset_default_graph()
    class_label = ['norm', 'defect1', 'defect2', 'defect3', 'defect4', 'defect5',
                   'defect6', 'defect7', 'defect8', 'defect9', 'defect10', 'defect11']
    img_index, res = [], []

    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])

    # 模型保存路径，前向传播
    if model == 'Alex':
        log_dir = "../log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=True,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)        # 输入不是规定的尺寸时，需要global_pool
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

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 恢复模型权重
        print("Reading checkpoints...")
        # ckpt 有 model_checkpoint_path 和 all_model_checkpoint_paths 两个属性
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('Error: no checkpoint file found')
            return

        img_list = os.listdir(test_dir)
        for img_name in img_list:
            # print(img_name)
            # 读取图片并缩放
            img = Image.open(os.path.join(test_dir, img_name))
            img = img.resize((IMG_SIZE, IMG_SIZE))

            # 图片转为numpy的数组
            img = np.array(img, np.float32)
            img = np.expand_dims(img, axis=0)

            predictions = sess.run(y, feed_dict={x: img})
            pre = np.argmax(predictions)
            img_index.append(img_name)
            res.append(class_label[int(pre)])

        print('image: ', img_index, '\n', 'class:', res)
        # 存为csv文件。两列：图片名，预测结果
        data = pd.DataFrame({'0': img_index, '1': res})
        # data.sort_index(axis=0)
        data.to_csv("../submit/vgg_2.csv", index=False, header=False)


if __name__ == '__main__':
    submit('../data/guangdong_round1_test_b_20181009')

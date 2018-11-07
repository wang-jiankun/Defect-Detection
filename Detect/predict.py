"""
Detect 预测
author: 王建坤
date: 2018-8-10
"""
import numpy as np
import tensorflow as tf
from nets import alexnet, vgg, inception_v4, resnet_v2
from PIL import Image
import os

CLASSES = 8
IMG_SIZE = 224
GLOBAL_POOL = False


def predict(img_path, model='Alex'):
    """
    预测图片
    :param img_path: 图片路径
    :param model: 模型名
    :return: none
    """
    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])

    # 模型保存路径，前向传播
    if model == 'Alex':
        log_path = "../log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=True,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)        # 输入不是规定的尺寸时，需要global_pool
    elif model == 'VGG':
        log_path = "../log/VGG"
        y, _ = vgg.vgg_16(x,
                          num_classes=CLASSES,
                          is_training=True,
                          dropout_keep_prob=1.0,
                          spatial_squeeze=True,
                          global_pool=GLOBAL_POOL)
    elif model == 'Incep4':
        log_path = "../log/Incep4"
        y, _ = inception_v4.inception_v4(x, num_classes=CLASSES,
                                         is_training=True,
                                         dropout_keep_prob=1.0,
                                         reuse=None,
                                         scope='InceptionV4',
                                         create_aux_logits=True)
    elif model == 'Res':
        log_path = "../log/Res"
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

    y = tf.nn.softmax(y)
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

        img = read_img(img_path)
        predictions = sess.run(y, feed_dict={x: img})
        pre = np.argmax(predictions)
        print('prediction is:', '\n', predictions)
        print('predict class is:', pre)


def check_tensor_name():
    """
    查看模型的 tensor，在调用模型之后使用
    :return: none
    """
    import tensorflow.contrib.slim as slim
    variables_to_restore = slim.get_variables_to_restore()
    for var in variables_to_restore:
        print(var.name)


def read_img(img_path):
    """
    读取指定路径的图片
    :param img_path: 图片的路径
    :return: numpy array of image
    """
    img = Image.open(img_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, np.float32)
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == '__main__':
    predict('../data/alum/擦花/擦花20180830172305对照样本.jpg')


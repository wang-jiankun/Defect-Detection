"""
缺陷检测QT软件--预测图片
author: 王建坤
date: 2018-9-25
"""
import numpy as np
import tensorflow as tf
from nets import alexnet, mobilenet_v1
from PIL import Image
import time

CLASSES = 5
IMG_SIZE = 224
GLOBAL_POOL = False


def load_model(model='Mobile'):
    global x, y, sess
    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])

    # 模型保存路径，前向传播
    if model == 'Alex':
        log_path = 'weight/Alex'
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=False,         # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)  # 输入不是规定的尺寸时，需要global_pool
    elif model == 'Mobile':
        log_path = 'weight/Mobile'
        y, _ = mobilenet_v1.mobilenet_v1(x,
                                         num_classes=CLASSES,
                                         dropout_keep_prob=1.0,
                                         is_training=False,
                                         min_depth=8,
                                         depth_multiplier=1.0,
                                         conv_defs=None,
                                         prediction_fn=None,
                                         spatial_squeeze=True,
                                         reuse=None,
                                         scope='MobilenetV1',
                                         global_pool=GLOBAL_POOL)
    else:
        print('Error: model name not exist')
        return
    y = tf.nn.softmax(y)

    saver = tf.train.Saver()

    sess = tf.Session()
    # 恢复模型权重
    print('Reading checkpoints from: ', model)
    ckpt = tf.train.get_checkpoint_state(log_path)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('Error: no checkpoint file found')
        return -1


def close_sess():
    sess.close()
    tf.reset_default_graph()
    print('Close the session successfully')


def predict(img_path):
    img = Image.open(img_path)
    if img.mode != 'L':
        print('Error: the image format is not support')
        return -1, -1
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, np.float32)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    start_time = time.clock()
    predictions = sess.run(y, feed_dict={x: img})
    pre = np.argmax(predictions)
    end_time = time.clock()
    run_time = round(end_time - start_time, 3)
    print('Detection is done. class: %d running time: %s s ' % (int(pre), run_time))
    # print('prediction is:', '\n', predictions)
    return pre, run_time


if __name__ == '__main__':
    print('run predict:')
    # load_model()

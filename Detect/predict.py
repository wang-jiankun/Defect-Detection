"""
Detect 预测
author: 王建坤
date: 2018-8-10
"""
from Detect.config import *
from PIL import Image
import time

# 图像目录路径
# IMG_DIR = '../data/crop/pos/'
# IMG_DIR = '../data/phone/'
IMG_DIR = '../data/cigarette/template/'      # normal
IS_TRAINING = False


def predict(img_path, model=MODEL_NAME):
    """
    预测图片
    :param img_path: 图片路径
    :param model: 模型名
    :return: none
    """
    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, CHANNEL])

    # 模型保存路径，前向传播
    if model == 'Alex':
        log_dir = "../log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=IS_TRAINING,  # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)        # 输入不是规定的尺寸时，需要global_pool
    elif model == 'My':
        log_dir = "../log/My"
        y = mynet.mynet_v1(x, is_training=IS_TRAINING, num_classes=CLASSES)
    elif model == 'Mobile':
        log_dir = "../log/Mobile"
        y, _ = mobilenet_v1.mobilenet_v1(x,
                                         num_classes=CLASSES,
                                         dropout_keep_prob=1.0,
                                         is_training=IS_TRAINING,
                                         min_depth=8,
                                         depth_multiplier=1.0,
                                         conv_defs=None,
                                         prediction_fn=None,
                                         spatial_squeeze=True,
                                         reuse=None,
                                         scope='MobilenetV1',
                                         global_pool=GLOBAL_POOL)
    elif model == 'VGG':
        log_dir = "../log/VGG"
        y, _ = vgg.vgg_16(x,
                          num_classes=CLASSES,
                          is_training=IS_TRAINING,
                          dropout_keep_prob=1.0,
                          spatial_squeeze=True,
                          global_pool=GLOBAL_POOL)
    elif model == 'Incep4':
        log_dir = "../log/Incep4"
        y, _ = inception_v4.inception_v4(x, num_classes=CLASSES,
                                         is_training=IS_TRAINING,
                                         dropout_keep_prob=1.0,
                                         reuse=None,
                                         scope='InceptionV4',
                                         create_aux_logits=True)
    elif model == 'Res':
        log_dir = "../log/Res"
        y, _ = resnet_v2.resnet_v2_50(x,
                                      num_classes=CLASSES,
                                      is_training=IS_TRAINING,
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
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            # check_tensor_name(sess)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('Error: no checkpoint file found')
            return

        # img = read_img(img_path)
        # # 第一次运行时间会较长
        # sess.run(y, feed_dict={x: img})
        if img_path:
            img_list = [img_path]
        else:
            img_list = os.listdir(IMG_DIR)
        for img_name in img_list:
            # img_name = '1.jpg'
            print(img_name)
            img_path = IMG_DIR + img_name
            img = read_img(img_path)
            # 如果输入是灰色图，要增加一维
            if CHANNEL == 1:
                img = np.expand_dims(img, axis=3)

            start_time = time.clock()
            predictions = sess.run(y, feed_dict={x: img})
            pre = np.argmax(predictions, 1)
            end_time = time.clock()
            runtime = end_time - start_time
            print('prediction is:', predictions)
            print('predict class is:', pre)
            print('run time:', runtime)


def check_tensor_name(sess):
    """
    查看模型的 tensor，在调用模型之后使用
    :return: none
    """
    import tensorflow.contrib.slim as slim
    variables_to_restore = slim.get_variables_to_restore()
    for var in variables_to_restore:
        # tensor 的名
        print(var.name)
        # tensor 的值
        print(sess.run(var))


def read_img(img_path):
    """
    读取指定路径的图片
    :param img_path: 图片的路径
    :return: numpy array of image
    """
    img = Image.open(img_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == '__main__':
    predict('1.jpg')


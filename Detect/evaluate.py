"""
Detect 性能评估
author: 王建坤
date: 2018-9-10
"""
from Detect.config import *
import math
from sklearn.model_selection import train_test_split

BATCH_SIZE = 1
IS_TRAINING = False


def evaluate(model=MODEL_NAME):
    """
    评估模型
    :param model: model name
    :return: none
    """
    # 预测为某类的样本个数，某类预测正确的样本个数
    sample_labels = np.zeros(CLASSES, dtype=np.uint16)
    pre_pos = np.zeros(CLASSES, dtype=np.uint16)
    true_pos = np.zeros(CLASSES, dtype=np.uint16)

    # 加载测试集
    images = np.load(images_path)
    labels = np.load(labels_path)

    _, val_data, _, val_label = train_test_split(images, labels, test_size=0.2, random_state=222)
    # 如果输入是灰色图，要增加一维
    if CHANNEL == 1:
        val_data = np.expand_dims(val_data, axis=3)

    test_num = val_data.shape[0]
    # 存放预测结果
    res = np.zeros(test_num, dtype=np.uint16)

    # 占位符
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, CHANNEL], name="x_input")
    y_ = tf.placeholder(tf.uint8, [None], name="y_input")

    # 模型保存路径，模型名，预训练文件路径，前向传播
    if model == 'Alex':
        log_dir = "../log/Alex"
        y, _ = alexnet.alexnet_v2(x,
                                  num_classes=CLASSES,      # 分类的类别
                                  is_training=IS_TRAINING,  # 是否在训练
                                  dropout_keep_prob=1.0,    # 保留比率
                                  spatial_squeeze=True,     # 压缩掉1维的维度
                                  global_pool=GLOBAL_POOL)  # 输入不是规定的尺寸时，需要global_pool
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
        log_dir = "E:/alum/log/Incep4"
        y, _ = inception_v4.inception_v4(x, num_classes=CLASSES,
                                         is_training=IS_TRAINING,
                                         dropout_keep_prob=1.0,
                                         reuse=None,
                                         scope='InceptionV4',
                                         create_aux_logits=True)
    elif model == 'Res':
        log_dir = "E:/alum/log/Res"
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
        num_iter = int(math.ceil(test_num / BATCH_SIZE))

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
            pres, pre = sess.run([y, y_pre], feed_dict={x: image_batch, y_: label_batch})
            res[start:end] = pre
            start += BATCH_SIZE
            step += 1
    # 计算准确率
    normal_num = 0
    for i in range(test_num):
        pre_pos[res[i]] += 1
        sample_labels[val_label[i]] += 1
        if res[i] == val_label[i]:
            true_pos[res[i]] += 1
        if val_label[i] == 0:
            normal_num += 1

    precision = true_pos/pre_pos
    recall = true_pos/sample_labels
    f1 = 2*precision*recall/(precision+recall)
    error = (pre_pos - true_pos) / (test_num - sample_labels)
    print('测试样本数：', test_num)
    print('测试数：', sample_labels)
    print('预测数：', pre_pos)
    print('正确数：', true_pos)
    print('Precision：', precision)
    print('Recall：', recall)
    print('F1：', f1)
    print('误检率：', error)
    print('准确率：', np.sum(true_pos)/test_num)
    print('总漏检率：', (pre_pos[0]-true_pos[0])/(test_num-normal_num))
    print('总过杀率：', (normal_num-true_pos[0])/normal_num)


if __name__ == '__main__':
    evaluate()

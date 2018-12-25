"""
Siamese 训练 -- 相似度损失函数
author: 王建坤
date: 2018-9-30
"""
from sklearn.model_selection import train_test_split
from Siamese import inference, utils
from Siamese.config import *

MAX_STEP = 2000
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.98
# 训练信息和保存权重的gap
INFO_STEP = 20
SAVE_STEP = 200
# 图像尺寸
BATCH_SIZE = 16


def train(inherit=False):
    # 加载数据集
    images = np.load(images_path)
    labels = np.load(labels_path)
    train_data, val_data, train_label, val_label = train_test_split(images, labels, test_size=0.2, random_state=222)
    # 如果输入是灰色图，要增加一维
    if CHANNEL == 1:
        train_data = np.expand_dims(train_data, axis=3)
        val_data = np.expand_dims(val_data, axis=3)

    # 占位符
    x1 = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, CHANNEL], name="x_input1")
    x2 = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, CHANNEL], name="x_input2")
    y = tf.placeholder(tf.float32, [None], name="y_input")
    # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    my_global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    # 前向传播
    with tf.variable_scope('siamese') as scope:
        out1, _ = mobilenet_v1.mobilenet_v1(x1,
                                            num_classes=CLASSES,
                                            dropout_keep_prob=1.0,
                                            is_training=True,
                                            min_depth=8,
                                            depth_multiplier=1.0,
                                            conv_defs=None,
                                            prediction_fn=None,
                                            spatial_squeeze=True,
                                            reuse=tf.AUTO_REUSE,
                                            scope='MobilenetV1',
                                            global_pool=GLOBAL_POOL)
        # 参数共享，不会生成两套参数。注意定义variable时要使用get_variable()
        # scope.reuse_variables()
        out2, _ = mobilenet_v1.mobilenet_v1(x2,
                                            num_classes=CLASSES,
                                            dropout_keep_prob=1.0,
                                            is_training=True,
                                            min_depth=8,
                                            depth_multiplier=1.0,
                                            conv_defs=None,
                                            prediction_fn=None,
                                            spatial_squeeze=True,
                                            reuse=tf.AUTO_REUSE,
                                            scope='MobilenetV1',
                                            global_pool=GLOBAL_POOL)

    # 损失函数和优化器
    with tf.variable_scope('metrics') as scope:
        loss = inference.loss_spring(out1, out2, y)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, my_global_step, 100, LEARNING_RATE_DECAY)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=my_global_step)

    saver = tf.train.Saver(tf.global_variables())

    # 模型保存路径和名称
    log_dir = "../log/Siamese"
    model_name = 'siamese.ckpt'

    with tf.Session() as sess:
        step = 0
        if inherit:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Siamese continue train from %s' % global_step)
                step = int(global_step)
            else:
                print('No checkpoint file found')
                return
        else:
            print('restart train')
            sess.run(tf.global_variables_initializer())

        while step < MAX_STEP:
            # 获取一对batch的数据集
            xs_1, ys_1 = utils.get_batch(train_data, train_label, BATCH_SIZE)
            xs_2, ys_2 = utils.get_batch(train_data, train_label, BATCH_SIZE)
            # 判断对应的两个标签是否相等
            y_s = np.array(ys_1 == ys_2, dtype=np.float32)

            _, y1, y2, train_loss = sess.run([train_op, out1, out2, loss],
                                             feed_dict={x1: xs_1, x2: xs_2, y: y_s})

            # 训练信息和保存模型
            step += 1
            if step % INFO_STEP == 0 or step == MAX_STEP:
                print('step: %d, loss: %.4f' % (step, train_loss))

            if step % SAVE_STEP == 0 or step == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, model_name)
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()

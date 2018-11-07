"""
Siamese 预测
author: 王建坤
date: 2018-8-16
"""
import tensorflow as tf
import numpy as np
from Siamese import inference

# 搭建图和恢复模型
# 占位符
with tf.variable_scope('input_x1') as scope:
    x1 = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])
with tf.variable_scope('input_x2') as scope:
    x2 = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])

# 前向传播
with tf.variable_scope('siamese') as scope:
    out1 = inference.inference(x1, 1)
    # 参数共享，不会生成两套参数
    scope.reuse_variables()
    out2 = inference.inference(x2, 1)

diff = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2)))

saver = tf.train.Saver(tf.global_variables())
# 模型保存路径
log_dir = "E:/alum/log/Siamese"

sess = tf.Session()

# 重复使用变量
# tf.get_variable_scope().reuse_variables()

print("Reading checkpoints...")
ckpt = tf.train.get_checkpoint_state(log_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading success, global_step is %s' % global_step)
else:
    print('No checkpoint file found')


# 单独预测，不用重复加载模型，sess是全局变量
def predict(img1, img2):
    x_1 = np.expand_dims(img1, axis=3)
    x_2 = np.expand_dims(img2, axis=3)
    y1, y2, diff_loss = sess.run([out1, out2, diff], feed_dict={x1: x_1, x2: x_2})
    print(y1, '\n', y2)
    print(diff_loss)

# 加载模型和预测在一个函数中
# def predict(img1, img2):
#     x_1 = np.expand_dims(img1, axis=3)
#     x_2 = np.expand_dims(img2, axis=3)
#
#     # 占位符
#     with tf.variable_scope('input_x1') as scope:
#         x1 = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])
#     with tf.variable_scope('input_x2') as scope:
#         x2 = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])
#
#     # 前向传播
#     with tf.variable_scope('siamese') as scope:
#         out1 = inference.inference(x1, 1)
#         # 参数共享，不会生成两套参数
#         scope.reuse_variables()
#         out2 = inference.inference(x2, 1)
#
#     diff = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2)))
#
#     saver = tf.train.Saver(tf.global_variables())
#     # 模型保存路径
#     log_dir = "E:/alum/log/Siamese"
#
#     with tf.Session() as sess:
#         tf.get_variable_scope().reuse_variables()
#         print("Reading checkpoints...")
#         ckpt = tf.train.get_checkpoint_state(log_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#             saver.restore(sess, ckpt.model_checkpoint_path)
#             print('Loading success, global_step is %s' % global_step)
#         else:
#             print('No checkpoint file found')
#             return
#
#         # 查看图的 tensor
#         # for var in tf.global_variables():
#         #     print(var.name)
#
#         # 查看 tensor 的值
#         # print(sess.run(tf.get_default_graph().get_tensor_by_name("siamese/b2:0")))
#
#         y1, y2, diff_loss = sess.run([out1, out2, diff], feed_dict={x1: x_1, x2: x_2})
#         # print(y1, '\n', y2)
#         print(diff_loss)


if __name__ == '__main__':
    images = np.load('E:/dataset/npy/train_data.npy')
    labels = np.load('E:/dataset/npy/train_label.npy')
    index1 = 0
    img1 = images[index1:index1+1]
    index2 = 1
    img2 = images[index2:index2+1]
    print(img1.shape)
    predict(img1, img2)

    # for index2 in range(labels.shape[0]):
    #     img2 = images[index2:index2 + 1]
    #     predict(img1, img2)

    # 两个样本间的关系
    # correct = np.equal(np.argmax(labels[index1]), np.argmax(labels[index2]))
    # print('true relationship is: ', correct)


#     first = 0
#     second = 0
#     third = 0
#     for i in range(120):
#         for j in range(240, 400):
#             ou1, ou2 = predict(images[i:i+1], images[j:j+1])
#             # ou2 = predict(images[301:302], images[301:302])
#             diffe = np.sqrt(np.sum(np.square(ou1 - ou2), 1))
#             # print('difference is: ', diffe)
#             if diffe < 1:
#                 first+=1
#             elif 1<= diffe <5:
#                 second +=1
#             else:
#                 third += 1
#     print(first, second, third)

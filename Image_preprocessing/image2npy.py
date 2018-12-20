"""
图片数据集转 npy 文件
author: 王建坤
date: 2018-10-15
"""
import numpy as np
import os
from PIL import Image, ImageFile
from matplotlib import pyplot as plt

# 读图出错的解决方法
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_SIZE = 224    # 299


def phone_to_npy(data_path, save_path):
    """
    手机壳图片数据集保存为 npy 文件
    :param data_path: the root path of data set
    :param save_path: the save path of npy
    :return: none
    """
    data, label = [], []
    per_class_num = {}

    # 图片类型
    class_dic = {'pos': 0, 'neg': 1}
    # 遍历文件夹
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        if folder not in class_dic.keys():
            continue
        # 遍历文件夹中的文件
        num = 0
        class_path = os.path.join(data_path, folder)
        img_list = os.listdir(class_path)
        for img_name in img_list:
            print(img_name)
            # 读取图片
            img = Image.open(os.path.join(class_path, img_name))
            # 添加数据和标签，根据文件夹名确定样本的标签
            img = np.array(img)
            data.append(img)
            label.append(class_dic[folder])
            num += 1
        per_class_num[folder] = num

    # 数据集转化为numpy数组
    data = np.array(data, np.uint8)
    label = np.array(label, np.uint8)
    print('per class number is: ', per_class_num)
    print('data set shape is: ', np.shape(data), np.shape(label))

    # 数组保存为npy
    data_save_path = save_path+'/'+'phone_data' + '.npy'
    label_save_path = save_path + '/' + 'phone_label' + '.npy'
    np.save(data_save_path, data)
    np.save(label_save_path, label)


def alum_to_npy(data_path, save_path):
    """
    铝型材图片数据集保存为 npy 文件
    :param data_path: the root path of data set
    :param save_path: the save path of npy
    :return: none
    """
    data, label = [], []
    per_class_num = {}
    save_name = 'cig_'

    # 图片类型
    # class_dic = {'正常': 0, '不导电': 1, '擦花': 2, '角位漏底': 3, '桔皮': 4, '漏底': 5, '起坑': 6, '脏点': 7}
    class_dic = {'normal': 0, 'nothing': 1, 'lack_cotton': 2, 'lack_piece': 3, 'wire_fail': 4}
    # 遍历文件夹
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        if folder not in class_dic.keys():
            continue

        # 遍历文件夹中的文件
        num = 0
        class_path = os.path.join(data_path, folder)
        img_list = os.listdir(class_path)
        for img_name in img_list:
            print(img_name)
            # 读取图片并缩放
            img = Image.open(os.path.join(class_path, img_name))
            img = img.resize((IMG_SIZE, IMG_SIZE))
            # 添加数据和标签，根据文件夹名确定样本的标签
            img = np.array(img)
            data.append(img)
            label.append(class_dic[folder])
            num += 1
        per_class_num[folder] = num

    # 数据集转化为numpy数组
    data = np.array(data, np.uint8)
    label = np.array(label, np.uint8)
    print('per class number is: ', per_class_num)
    print('data set shape is: ', np.shape(data), np.shape(label))

    # 数组保存为npy
    data_save_path = save_path+'/'+save_name+'data_'+str(IMG_SIZE)+'.npy'
    label_save_path = save_path+'/'+save_name+'label_'+str(IMG_SIZE)+'.npy'
    np.save(data_save_path, data)
    np.save(label_save_path, label)


def load_npy(data_path, label_path):
    """
    读取 npy 文件
    :param data_path:
    :param label_path:
    :return:
    """
    data = np.load(data_path)
    label = np.load(label_path)
    print('data set shape is: ', np.shape(data))
    return data, label


def array_to_image(sample, sample_label):
    """
    显示一个样本的信息,图片和标签
    :param sample:
    :param sample_label:
    :return:
    """
    print('sample shape: ', np.shape(sample))
    sample = sample/255
    # plt.imshow 的像素值为 [0, 1]
    plt.imshow(sample)
    plt.show()
    print('sample label: ', sample_label)


if __name__ == '__main__':
    print('running image2npy:')
    # alum_to_npy('../data/alum', '../data')
    alum_to_npy('../data/cigarette', '../data')
    # phone_to_npy('../data/crop', '../data')




"""
工具函数
author: 王建坤
date: 2018-10-15
"""
import numpy as np
import os
import pandas as pd


def load_npy(data_path, label_path):
    """
    Load npy file
    :param data_path: the path of data directory
    :param label_path: the path of label directory
    :return: numpy format of data and label
    """
    data = np.load(data_path)
    label = np.load(label_path)
    print('data_set shape: ', np.shape(data))
    return data, label


def get_batch(data, label, batch_size):
    """
    Get a batch from numpy array.
    :param data: numpy array of data
    :param label: numpy array of label
    :param batch_size: the number of a batch
    :param total: the number of data set sample
    :return: a batch of data set
    """
    # 遍历数据集
    # start = start % total
    # end = start + batch_size
    # if end > total:
    #     res = end % total
    #     end = total
    #     data1 = data[start:end]
    #     data2 = data[0: res]
    #     return np.vstack((data1, data2))
    # return data[start:end]

    # 随机选取一个batch
    total = data.shape[0]
    index_list = np.random.randint(total, size=batch_size)
    return data[index_list, :], label[index_list]


def shuffle_data(data, label):
    """
    Shuffle the data to assure that the training model is valid.
    """
    permutation = np.random.permutation(label.shape[0])
    batch_data = data[permutation, :]
    batch_label = label[permutation]
    return batch_data, batch_label


def normalize_data(data):
    """
    Normalized the data by subtracting the mean value of R,G,B to accelerate training .
    """
    r = data[:, :, :, 2]
    r_mean = np.average(r)
    b = data[:, :, :, 0]
    b_mean = np.average(b)
    g = data[:, :, :, 1]
    g_mean = np.average(g)
    r = r - r_mean
    b = b - b_mean
    g = g - g_mean
    data = np.zeros([r.shape[0], r.shape[1], r.shape[2], 3])
    data[:, :, :, 0] = b
    data[:, :, :, 1] = g
    data[:, :, :, 2] = r
    return data


def check_tensor():
    """
    查看保存的 ckpt文件中的 tensor
    """
    from tensorflow.python import pywrap_tensorflow

    # ckpt 文件路径
    checkpoint_path = 'E:/alum/weight/inception_v4_2016_09_09/inception_v4.ckpt'

    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Print tensor name and values
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key).dtype)


def image_to_csv(data_path, save_path):
    """
    图片路径和标签保存为 csv
    :param data_path: image root path
    :param save_path: save path of csv file
    :return: none
    """
    img_path, label = [], []
    per_class_num = {}
    # 图片类型
    class_dic = {'正常': 0, '不导电': 1, '擦花': 2, '角位漏底': 3, '桔皮': 4, '漏底': 5, '起坑': 6, '脏点': 7}
    # 遍历文件夹
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        if folder not in class_dic.keys():
            continue

        num = 0
        class_path = os.path.join(data_path, folder)
        img_list = os.listdir(class_path)
        for img_name in img_list:
            img_path.append(os.path.join(class_path, img_name))
            label.append(class_dic[folder])
            num += 1
        per_class_num[folder] = num

    img_path_label = pd.DataFrame({'img_path': img_path, 'label': label})
    img_path_label.to_csv(save_path+'/data_label.csv', index=False)
    print('per class number: ', per_class_num)


def csv_show_image():
    """
    从 csv 文件读取图片路径，并显示
    :return: none
    """
    from PIL import Image
    table = pd.read_csv('../data/data_label.csv')
    img_path = table.iloc[0, 0]
    img = Image.open(img_path)
    img.show()


if __name__ == "__main__":
    print('running utils:')
    # train_data, train_label = load_npy('../data/data_299.npy', '../data/label_299.npy')
    # image_to_csv('../data/alum', '../data')



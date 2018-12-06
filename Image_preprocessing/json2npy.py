"""
把 json文件转为数据集，并保存为npy
author: 王建坤
date: 2018-10-26
"""
# labelme_json_to_dataset E:/1.json
# labelme_draw_json E:/1.json
import json
import os
import numpy as np
from PIL import Image

IMG_SIZE = 224
col = 2592
row = 1944


def json_to_npy(root_path, save_path='../data/'):
    """
    把 json文件转为数据集，并保存为npy
    :param root_path:
    :param save_path:
    :return:
    """
    data, labels = [], []
    f = open('../data/label.txt', 'a')
    file_list = os.listdir(root_path)
    for file_name in file_list:
        if file_name.split('.')[-1] == 'jpg':
            img = Image.open(os.path.join(root_path, file_name))
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img)
            img = np.expand_dims(img, 3)
            data.append(img)
            json_path = os.path.join(root_path, file_name.split('.')[0]) + '.json'
            label = json_to_label(json_path)
            f.write(str(label) + '\n')
            temp = [label[0][0]/col, label[0][1]/row, label[1][0]/col, label[1][1]/row,
                    label[2][0]/col, label[2][1]/row, label[3][0]/col, label[3][1]/row]
            labels.append(temp)

    f.close()
    # 数据集转化为numpy数组
    data = np.array(data, np.uint8)
    labels = np.array(labels, np.float16)
    print('data set shape is: ', np.shape(data), np.shape(labels))

    # 数组保存为npy
    data_save_path = save_path + 'card_data.npy'
    label_save_path = save_path + 'card_label.npy'
    np.save(data_save_path, data)
    np.save(label_save_path, labels)


def json_to_label(json_path):
    """
    抽取 json 文件中的 label
    :param json_path:
    :return: 
    """
    with open(json_path, 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
        label = data['shapes'][0]['points']
        # print(label)
        return label


if __name__ == '__main__':
    print('running json2npy: ')
    json_to_npy('../data/card')
    # json_to_label('E:/1.json')



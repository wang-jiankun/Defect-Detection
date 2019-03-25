"""
电子烟雾化器装配检测，效果评估
author: 王建坤
date: 2019-3-6
"""
import os
from mv import cig_detection
import numpy as np

ROOT_DIR = 'E:/backup/cigarette/'
CLASSES = 5


def evaluate():
    """

    :return:
    """
    img_path_list, labels = [], []

    class_dic = {'normal': 0, 'nothing': 1, 'lack_cotton': 2, 'lack_piece': 3, 'wire_fail': 4}
    # 遍历文件夹
    folder_list = os.listdir(ROOT_DIR)
    for folder in folder_list:
        # 过滤不需要的文件夹
        if folder not in class_dic.keys():
            continue
        folder_path = os.path.join(ROOT_DIR, folder)
        img_list = os.listdir(folder_path)
        for img_name in img_list:
            img_path = os.path.join(folder_path, img_name)
            img_path_list.append(img_path)
            labels.append(class_dic[folder])

    res = []
    for img, label in zip(img_path_list, labels):
        print(img)
        detector = cig_detection.AssembleDetection(img)
        detector.detect()
        if sum(detector.res) == 0:
            res.append(0)
        else:
            for i in range(len(detector.res)):
                if detector.res[i] == 1:
                    if i < 2:
                        res.append(i+1)
                        break
                    elif i == 2:
                        res.append(4)
                        break
                    else:
                        res.append(i)
                        break

    sample_labels = np.zeros(CLASSES, dtype=np.uint16)
    pre_pos = np.zeros(CLASSES, dtype=np.uint16)
    true_pos = np.zeros(CLASSES, dtype=np.uint16)

    test_num = len(res)
    normal_num = 0
    for i in range(test_num):
        pre_pos[res[i]] += 1
        sample_labels[labels[i]] += 1
        if res[i] == labels[i]:
            true_pos[res[i]] += 1
        if labels[i] == 0:
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



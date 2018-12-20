"""
电子烟雾化器装配检测，图像处理
author: 王建坤
date: 2018-12-18
"""
import cv2
import numpy as np
import time
import os
from mv import base

IMG_H_SIZE = 500
IMG_V_SIZE = 300
THRESH = 240
IMG_DIR = '../data/cigarette/normal/'
SIZE_RATIO = 50


def find_object():
    """
    确定工件的位置
    :return:
    """
    # 打开图片
    img_list = os.listdir(IMG_DIR)
    img_name = img_list[0]
    print('image name:', img_name)
    gray_img = base.open_img(os.path.join(IMG_DIR, img_name))
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    start_time = time.clock()

    # 二值化
    thresh_img = base.img_thresh(gray_img, THRESH, 255)
    # 开运算
    # thresh_img = base.img_morphology(thresh_img, 2, 21)
    # print(np.sum(thresh_img, 0))
    # box = base.find_object(thresh_img, color_img, 0, True)
    # if not box:
    #     print('No find object!')
    #     return
    # x, y, w, h = box
    # print(box)
    # center_roi = {'x1': int(x+2/5*w), 'y1': int(y+2/5*h), 'x2': int(x+3/5*w), 'y2': int(y+3/5*h)}
    # cv2.rectangle(color_img, (center_roi['x1'], center_roi['y1']), (center_roi['x2'], center_roi['y2']), (0, 0, 255), 3)
    # center_img = gray_img[center_roi['y1']:center_roi['y2'], center_roi['x1']:center_roi['x2']]
    # cv2.imshow('center_img', center_img)
    #
    # end_time = time.clock()
    # print('run time: ', end_time - start_time)

    # 显示图片
    cv2.imshow('src', color_img)
    cv2.imshow('thresh', thresh_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    find_object()



"""
图像裁剪，从大图片中裁剪出小图片
author: 王建坤
date: 2018-12-6
"""
import cv2
import os

IMG_H_SIZE = 2000
IMG_V_SIZE = 1000

IMG_DIR = '../data/phone'
SAVE_DIR = '../data/crop'


def open_img(img_path):
    """
    打开指定路径的单张图像（灰度图），并缩放
    :param img_path: 图像路径
    :return:
    """
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (IMG_H_SIZE, IMG_V_SIZE))
    return img


def crop_img(img):
    for x in range(20, 101, 8):
        for y in range(60, 701, 8):
            print(x, '-', y)
            small_image = img[x:x+128, y:y+128]
            cv2.imwrite(os.path.join(SAVE_DIR, str(x)+'_'+str(y)+'.jpg'), small_image)


def main():
    img_list = os.listdir(IMG_DIR)
    img_name = img_list[0]
    print('image name:', img_name)
    img = open_img(os.path.join(IMG_DIR, img_name))
    crop_img(img)


if __name__ == '__main__':
    main()

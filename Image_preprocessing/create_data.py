"""
生成手机后壳的虚假数据集--只是用来测试算法的效果
author: 王建坤
date: 2018-8-15
"""
import cv2
import numpy as np
import os
from PIL import ImageEnhance, Image


def img_to_gray(img_path):
    """
    彩色图片转为灰度图存储
    """
    # src_img = cv2.imread(img_path, 0)
    # print(src_img.shape)
    # cv2.imwrite('F:/DefectDetection/img_data/zc.jpg', src_img)
    # # cv2.cvtColor(src_img, gray_img, cv2.COLOR_BGR2GRAY);

    src_img = Image.open(img_path)
    # 显示图片的尺寸,图片的模式,图片的格式
    print(src_img.size, src_img.mode, src_img.format)
    dst_img = src_img.convert("L")
    # dst_img.save(os.path.join(os.path.dirname(img_path), 'zc.jpg'))
    dst_img.save(os.path.join(os.path.dirname(img_path), 'wz.jpg'))


def img_enhance(img_dir, save_path, num):
    """
    随机扰动图像的亮度和对比度
    """
    # img_path_list = [img_dir]
    img_path_list = os.listdir(img_dir)
    for img_path in img_path_list:
        # src_img = Image.open(img_path)
        # filename_pre = img_path.split('/')[-1].split('.')[0]

        src_img = Image.open(os.path.join(img_dir, img_path))
        # print(src_img.mode, src_img.size)
        filename_pre = img_path.split('.')[0]

        for i in range(num):
            brightness_factor = np.random.randint(97, 103) / 100.0
            brightness_image = ImageEnhance.Brightness(src_img).enhance(brightness_factor)
            contrast_factor = np.random.randint(98, 102) / 100.0
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(contrast_factor)
            save_name = save_path + '/' + filename_pre + '_' + str(i) + '.jpg'
            contrast_image.save(save_name)


def draw_line(img_path, save_path, num):
    """
    随机画线
    """
    src_img = cv2.imread(img_path, -1)
    filename_pre = save_path.split('/')[-1]

    for i in range(num):
        dst_img = src_img.copy()
        x1, y1 = np.random.randint(80, dst_img.shape[1]-120), np.random.randint(70, dst_img.shape[0]-70)
        x2, y2 = x1, y1
        while x1-10 < x2 < x1+10:
            x2 = np.random.randint(x1-50, x1+50)
        while y1-10 < y2 < y1+10:
            y2 = np.random.randint(y1-50, y1+50)
        # print(x1, y1, x2, y2)
        thickness = np.random.randint(1, 2)
        pixel = np.random.randint(160, 200)
        cv2.line(dst_img, (x1, y1), (x2, y2), pixel, thickness)

        save_name = save_path + '/' + filename_pre + '_' + str(i) + '.jpg'
        cv2.imwrite(save_name, dst_img)


def draw_wz(img_path, save_path, num):
    """
    随机贴上污渍
    """
    src_img = cv2.imread(img_path, -1)
    filename_pre = save_path.split('/')[-1]

    for i in range(num):
        dst_img = src_img.copy()
        x1, y1 = np.random.randint(50, dst_img.shape[1] - 60), np.random.randint(40, dst_img.shape[0] - 40)
        # print(x1, y1)
        radius = np.random.randint(5, 20)
        pixel = np.random.randint(180, 200)
        cv2.circle(dst_img, (x1, y1), radius, pixel, -1)

        save_name = save_path + '/' + filename_pre + '_' + str(i) + '.jpg'
        cv2.imwrite(save_name, dst_img)


if __name__ == '__main__':
    print('create_data.py is running')
    # img_to_gray('F:/DefectDetection/img_data/src_zc.jpg')
    # img_enhance('F:/DefectDetection/img_data/zc.jpg', 'F:/DefectDetection/img_data/zc', 2000)
    # draw_line('F:/DefectDetection/img_data/zc.jpg', 'F:/DefectDetection/img_data/hh', 500)
    # img_enhance('F:/DefectDetection/img_data/hh', 'F:/DefectDetection/img_data/hh', 3)
    # img_to_gray('F:/DefectDetection/img_data/src_wz.jpg')
    # draw_wz('F:/DefectDetection/img_data/zc.jpg', 'F:/DefectDetection/img_data/wz', 500)
    # img_enhance('F:/DefectDetection/img_data/wz', 'F:/DefectDetection/img_data/wz', 3)

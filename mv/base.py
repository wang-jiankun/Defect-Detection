"""
图像处理，基础方法
author: 王建坤
date: 2018-12-4
"""
import cv2
import numpy as np
import time
import os

IMG_H_SIZE = 2000   # 2592
IMG_V_SIZE = 1000   # 1994
THRESH = 50     # 40
IMG_DIR = '../data/phone'


def open_img(img_path, resize=False):
    """
    打开指定路径的单张图像（灰度图），并缩放
    :param img_path: 图像路径
    :param resize: 是否要缩放图片
    :return:
    """
    img = cv2.imread(img_path, 0)
    if resize:
        img = cv2.resize(img, (IMG_H_SIZE, IMG_V_SIZE))
    return img


def img_gradient(img_gray):
    """
    计算图像的梯度
    :param img_gray: 灰度图
    :return:
    """
    x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(abs_x, 1, abs_y, 1, 0)
    # ret, thresh = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
    return dst


def img_thresh(img_gray, thresh, max_val):
    """
    图像二值化
    :param img_gray: 灰度图
    :param thresh: 阈值
    :param max_val: 二值中的高值
    :return:
    """
    # 普通阈值：cv2.THRESH_BINARY,自适应阈值：cv2.THRESH_OTSU, cv2.THRESH_TRUNC
    _, dst = cv2.threshold(img_gray, thresh, max_val, cv2.THRESH_BINARY)
    return dst


def img_filter(img, method=1, k=3):
    """
    图像滤波，1：均值滤波，2：中值滤波，3：高斯滤波
    :param img: 输入图片
    :param method: 滤波器类型
    :param k: 滤波器窗口大小
    :return:
    """
    if method == 1:
        img_blur = cv2.blur(img, (k, k))
    elif method == 2:
        img_blur = cv2.medianBlur(img, k)
    else:
        img_blur = cv2.GaussianBlur(img, (k, k), 0)

    return img_blur


def find_object(img_bin, img, draw=False):
    """
    查找目标物体的轮廓
    :param img_bin: 二值化图
    :param img: 原图
    :param draw: 图像显示标志位
    :return:
    """
    approximate_method = 1
    _, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # 轮廓是否要近似为最小外包矩形
    if approximate_method == 1:
        # rect = cv2.boundingRect(contours[0])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour = 0
        area_thresh = img.shape[0] * img.shape[1]/3
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_thresh:
                break
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        print(box)
        if draw:
            cv2.polylines(img, [box], False, 255, 1)
    else:
        if draw:
            cv2.drawContours(img, contours, -1, 255, 1)
        box = None
    return box


def perspective_transform(img, points, show=False):
    """
    目标的投影变换
    :param img: 原图
    :param points: 目标边界框坐标点集
    :param show:
    :return:
    """
    pts1 = np.float32(points)
    pts2 = np.float32([[0, IMG_V_SIZE], [0, 0], [IMG_H_SIZE, 0], [IMG_H_SIZE, IMG_V_SIZE]])
    transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, transform_matrix, (IMG_H_SIZE, IMG_V_SIZE))
    if show:
        cv2.imshow('trans:', dst)
    return dst


def img_compare():
    """
    目标图片比较，两张目标图片相减
    :return:
    """
    img1 = open_img(os.path.join(IMG_DIR, '1.jpg'))

    start_time = time.clock()
    img1 = img_gradient(img1)
    end_time = time.clock()
    print('run time: ', end_time - start_time)

    img2 = open_img(os.path.join(IMG_DIR, '2.jpg'))
    # img2 = img_gradient(img2)

    dst = cv2.absdiff(img1, img2)
    dst = img_thresh(dst, THRESH, 255)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('compare', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def seg_object(method=0):
    """
    分割出目标，0: 梯度法, 1: 阈值法
    :return:
    """
    # 打开图片
    img_list = os.listdir(IMG_DIR)
    img_name = img_list[-5]
    print('image name:', img_name)
    gray_img = open_img(os.path.join(IMG_DIR, img_name))

    start_time = time.clock()

    # 梯度->二值化
    if method == 0:
        thresh_img = img_thresh(gray_img, THRESH, 255)
    else:
        gradient_img = img_gradient(gray_img)
        cv2.imshow('gradient', gradient_img)
        thresh_img = img_thresh(gradient_img, THRESH, 255)
    box = find_object(thresh_img, gray_img)
    dst = perspective_transform(gray_img, box)
    # blur_img = img_filter(thresh_img, 1)

    end_time = time.clock()
    print('run time: ', end_time - start_time)

    # 保存，显示图片
    cv2.imwrite(os.path.join(IMG_DIR, '1.jpg'), dst)
    cv2.imshow('src', gray_img)
    cv2.imshow('thresh', thresh_img)
    # cv2.imshow('dst', blur_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('run cv.base: ')
    seg_object()
    # img_compare()

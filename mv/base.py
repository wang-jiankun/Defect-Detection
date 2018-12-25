"""
图像处理，基础方法
author: 王建坤
date: 2018-12-4
"""
import cv2
import numpy as np
import time
import os

OBJECT = 'cig'
if OBJECT == 'phone':
    IMG_H_SIZE = 2000   # 2592
    IMG_V_SIZE = 1000   # 1994
    THRESH = 50     # 40
    IMG_DIR = '../data/phone/'
    SIZE_RATIO = 3
elif OBJECT == 'cig':
    IMG_H_SIZE = 500
    IMG_V_SIZE = 300
    THRESH = 50
    IMG_DIR = '../data/cigarette/normal/'
    SIZE_RATIO = 3
else:
    IMG_DIR = '../data/ocr/Cam1/'


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


def resize_img():
    """
    缩放文件夹路径下的所有图像，并保存
    :return:
    """
    img_list = os.listdir(IMG_DIR)
    for img_name in img_list:
        print('image name:', img_name)
        gray_img = open_img(os.path.join(IMG_DIR, img_name), True)
        cv2.imwrite(os.path.join(IMG_DIR, img_name), gray_img)


def single_sobel(img_gray, x, y):
    """
    单方向 sobel 算子
    :param img_gray:灰度图
    :param x:
    :param y:
    :return:
    """
    sobel = cv2.Sobel(img_gray, cv2.CV_16S, x, y)
    abs_sobel = cv2.convertScaleAbs(sobel)
    return abs_sobel


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


def img_thresh(img_gray, thresh, max_val, method=cv2.THRESH_BINARY):
    """
    图像二值化
    :param img_gray: 灰度图
    :param thresh: 阈值
    :param max_val: 二值中的高值
    :param method: 方法
    :return:
    """
    # 普通阈值：cv2.THRESH_BINARY,自适应阈值：cv2.THRESH_OTSU, cv2.THRESH_TRUNC
    _, dst = cv2.threshold(img_gray, thresh, max_val, method)
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


def img_morphology(img_gray, method=0, kernel_size=3, kernel_type=0):
    """
    形态学操作
    :param img_gray: 灰度图
    :param method: 方法，0：腐蚀，1：膨胀，2：开运算，3：闭运算
    :param kernel_size: 窗口的大小
    :param kernel_type: 窗口的类型，0：矩形，1：椭圆形，2：交叉形
    :return:
    """
    if method == 0:
        # 腐蚀
        img_gray = cv2.erode(img_gray, (kernel_size, kernel_size), iterations=2)
    elif method == 1:
        # 膨胀
        img_gray = cv2.dilate(img_gray, (kernel_size, kernel_size), iterations=2)
    else:
        if kernel_type == 0:
            k_type = cv2.MORPH_RECT
        elif kernel_type == 1:
            k_type = cv2.MORPH_ELLIPSE
        else:
            k_type = cv2.MORPH_CROSS
        kernel = cv2.getStructuringElement(k_type, (kernel_size, kernel_size))
        if method == 2:
            # 开运算
            img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        else:
            # 闭运算
            img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    return img_gray


def find_object(img_bin, img, approximate_method=0, draw=False):
    """
    查找目标物体的轮廓
    :param img_bin: 二值化图
    :param img: 原图
    :param approximate_method: 轮廓近似方式，0：矩形，1：最小矩形，2：点集
    :param draw: 绘制边界框标志位
    :return:
    """
    _, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    box = None
    # 轮廓是近似方式，0：矩形，1：最小矩形，2：点集
    if approximate_method == 0:
        area_thresh = img.shape[0] * img.shape[1] / SIZE_RATIO
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_thresh:
                box = x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    elif approximate_method == 1:
        contour = 0
        area_thresh = img.shape[0] * img.shape[1] / SIZE_RATIO
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


def vertical_flip(image):
    """
    对图片进行水平垂直翻转
    """
    original_image = image
    flip_v = cv2.flip(original_image, -1)
    return flip_v


def image_enhance():
    """
    对图片进行数据增强，包括：水平垂直翻转
    """
    img_list = os.listdir(IMG_DIR)
    for img_name in img_list:
        print('image name:', img_name)
        gray_img = open_img(os.path.join(IMG_DIR, img_name))
        flip_img = vertical_flip(gray_img)
        img_name = 'v_' + img_name

        cv2.imwrite(os.path.join(IMG_DIR, img_name), flip_img)
        # cv2.imshow('flip_img', flip_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


def seg_object(method=0):
    """
    分割出目标，0:阈值法 , 1: 梯度法
    :return:
    """
    # 打开图片
    img_list = os.listdir(IMG_DIR + '0')
    img_name = img_list[3]
    print('image name:', img_name)
    gray_img = open_img(os.path.join(IMG_DIR + '0', img_name))

    start_time = time.clock()

    # 二值化
    if method == 0:
        thresh_img = img_thresh(gray_img, THRESH, 255)
    # 梯度->二值化
    else:
        gradient_img = img_gradient(gray_img)
        cv2.imshow('gradient', gradient_img)
        thresh_img = img_thresh(gradient_img, THRESH, 255)
    thresh_img = img_morphology(thresh_img, 2, 3)
    box = find_object(thresh_img, gray_img, 0, True)
    # dst = perspective_transform(gray_img, box, True)
    # blur_img = img_filter(thresh_img, 1)

    end_time = time.clock()
    print('run time: ', end_time - start_time)

    # 保存，显示图片
    # cv2.imwrite(os.path.join(IMG_DIR, '1.jpg'), dst)
    cv2.imshow('src', gray_img)
    cv2.imshow('thresh', thresh_img)
    # cv2.imshow('dst', blur_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cut_roi():
    """
    根据位置，裁剪出装配电子烟的 ROI，并保存
    """
    # 打开图片
    img_list = os.listdir(IMG_DIR+'2')
    i = 1000
    for img_name in img_list:
        print('image name:', img_name)
        gray_img = open_img(os.path.join(IMG_DIR+'2', img_name))

        # 两个工位的 ROI
        # 0
        # left = gray_img[1580:2780, 850:2850]
        # right = gray_img[1580:2780, 3800:5800]

        # 1
        # left = gray_img[1700:2900, 870:2870]
        # right = gray_img[1700:2900, 3820:5820]

        # 2
        left = gray_img[1540:2740, 870:2870]
        right = gray_img[1540:2740, 3840:5840]

        cv2.imshow('l', left)
        cv2.imshow('r', right)
        img1_name = str(i) + '.jpg'
        img2_name = str(i+1) + '.jpg'
        i += 2
        # cv2.imwrite(os.path.join(IMG_DIR, img1_name), left)
        # cv2.imwrite(os.path.join(IMG_DIR, img2_name), right)

        cv2.waitKey()
        cv2.destroyAllWindows()


def test():
    """
    试验
    """
    img_list = os.listdir(IMG_DIR)
    img_name = img_list[-1]
    print('image name:', img_name)
    gray_img = open_img(os.path.join(IMG_DIR, img_name), True)
    # gray_img = gray_img[105:205, 160:440]
    # gray_img = 255 - gray_img
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # thresh_img = img_thresh(gray_img, 250, 0, cv2.THRESH_BINARY)
    # print(np.sum(thresh_img))
    # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
    # dst = img_morphology(thresh_img, 5, 3)
    dst = single_sobel(gray_img, 0, 1)
    dst = img_gradient(gray_img)

    cv2.imshow('src', gray_img)
    # cv2.imshow('thresh', thresh_img)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('run cv.base: ')
    # seg_object()
    # cut_roi()
    test()
    # image_enhance()

"""
电子烟雾化器装配检测，图像处理
author: 王建坤
date: 2018-12-18
"""
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mv import base

IMG_H_SIZE = 2000
IMG_V_SIZE = 1200
THRESH = 55
IMG_DIR = '../data/cigarette/lack_piece/'
# normal, nothing, lack_cotton, lack_piece, wire_fail
TEMPL_DIR = 'E:/Defect_Detection/data/cigarette/template/'


def open_img(img_path, resize=False):
    """
    打开指定路径的单张图像（灰度图），并缩放
    :param img_path: 图像路径
    :param resize: 是否要缩放图片
    :return:
    """
    img = cv2.imread(img_path, 0)
    if resize:
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    return img


def find_object():
    """
    确定工件的位置
    :return:
    """
    # 打开图片
    img_list = os.listdir(IMG_DIR)
    img_name = img_list[0]
    print('image name:', img_name)
    gray_img = open_img(os.path.join(IMG_DIR, img_name), True)
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    start_time = time.clock()

    # 二值化
    thresh_img = base.img_thresh(gray_img, THRESH, 255)
    # 开运算
    thresh_img = base.img_morphology(thresh_img, 2, 21)
    # print(np.sum(thresh_img, 0))
    box = base.find_object(thresh_img, color_img, 0, True)
    if not box:
        print('No find object!')
        return
    x, y, w, h = box
    print(box)

    # 正中间
    center_roi = {'x1': int(x+2/5*w), 'y1': int(y+2/5*h), 'x2': int(x+3/5*w), 'y2': int(y+3/5*h)}
    draw_roi(color_img, center_roi)
    center_img = roi_img(gray_img, center_roi)
    cv2.imshow('center_img', center_img)
    # 左中
    lc_roi = {'x1': int(x+1/8*w), 'y1': int(y+3/7*h), 'x2': int(x+1/4*w), 'y2': int(y+4/7*h)}
    draw_roi(color_img, lc_roi)
    lc_img = roi_img(gray_img, lc_roi)
    cv2.imshow('lc_img', lc_img)
    # 右中
    rc_roi = {'x1': int(x+3/4*w), 'y1': int(y+3/7*h), 'x2': int(x+7/8*w), 'y2': int(y+4/7*h)}
    draw_roi(color_img, rc_roi)
    rc_img = roi_img(gray_img, rc_roi)
    cv2.imshow('rc_img', rc_img)
    # 左边
    # left_roi = {'x1': int(x-w/10), 'y1': int(y), 'x2': int(x-10), 'y2': int(y+h)}
    # draw_roi(color_img, left_roi)
    # left_img = roi_img(gray_img, left_roi)
    # cv2.imshow('left_img', left_img)
    # 顶部
    top_roi = {'x1': int(x+w*3/5), 'y1': int(y-h/6), 'x2': int(x+w*7/10), 'y2': int(y-10)}
    draw_roi(color_img, top_roi)
    top_img = roi_img(gray_img, top_roi)
    cv2.imshow('top_img', top_img)
    # 底部
    bottom_roi = {'x1': int(x+w*3/10), 'y1': int(y+h+10), 'x2': int(x+w*2/5), 'y2': int(y+h*7/6)}
    draw_roi(color_img, bottom_roi)
    bottom_img = roi_img(gray_img, bottom_roi)
    cv2.imshow('bottom_img', bottom_img)
    # 左上
    lt_roi = {'x1': int(x+w*1/4), 'y1': int(y+h*3/20), 'x2': int(x+w*2/5), 'y2': int(y+h*1/3)}
    draw_roi(color_img, lt_roi)
    lt_img = roi_img(gray_img, lt_roi)
    cv2.imshow('lt_img', lt_img)
    # 左下
    lb_roi = {'x1': int(x+w*1/4), 'y1': int(y+h*2/3), 'x2': int(x+w*2/5), 'y2': int(y+h*17/20)}
    draw_roi(color_img, lb_roi)
    lb_img = roi_img(gray_img, lb_roi)
    cv2.imshow('lb_img', lb_img)
    # 右上
    rt_roi = {'x1': int(x+w*3/5), 'y1': int(y+h*3/20), 'x2': int(x+w*3/4), 'y2': int(y+h*1/3)}
    draw_roi(color_img, rt_roi)
    rt_img = roi_img(gray_img, rt_roi)
    cv2.imshow('rt_img', rt_img)
    # 右下
    rb_roi = {'x1': int(x+w*3/5), 'y1': int(y+h*2/3), 'x2': int(x+w*3/4), 'y2': int(y+h*17/20)}
    draw_roi(color_img, rb_roi)
    rb_img = roi_img(gray_img, rb_roi)
    cv2.imshow('rb_img', rb_img)

    # roi 处理
    roi_c = lt_img
    roi_c = template_pre(roi_c)
    roi_c = base.img_thresh(roi_c, 240, 1)
    templ_img = open_img(TEMPL_DIR + 'lt.jpg', True)
    templ_img = base.img_thresh(templ_img, 240, 0)
    cv2.imshow('templ_img', templ_img)
    dst = template_match(roi_c, templ_img, 0)
    print(np.min(dst))

    cv2.imshow('roi_c', roi_c)
    v_sum = np.sum(roi_c, 0, dtype=np.float)
    v_sum[v_sum < 6] = 0
    v_sum = np.convolve(v_sum, [0.2, 0.2, 0.2, 0.2, 0.2])
    plt.plot(v_sum)
    plt.show()
    print(v_sum)
    print(np.gradient(v_sum))

    end_time = time.clock()
    print('run time: ', end_time - start_time)

    # 显示图片
    cv2.imshow('src', color_img)
    cv2.imshow('thresh', thresh_img)
    # cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def roi_img(img, roi):
    """
    截取图片的指定 ROI
    :param img:
    :param roi:
    :return:
    """
    return img[roi['y1']:roi['y2'], roi['x1']:roi['x2']]


def draw_roi(color_img, roi):
    """
    绘制 ROI 框
    :param color_img:
    :param roi:
    :return:
    """
    cv2.rectangle(color_img, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (0, 0, 255), 2)


def template_pre(img):
    """
    模板预处理
    :param img:
    :return:
    """
    # 闭运算
    temp_img = base.img_morphology(img, 3, 7)
    return temp_img


def template_match(image, templ, method=0):
    """
    :param image:
    :param templ:
    :param method:
    :return:
    """
    if method == 0:
        method = cv2.TM_SQDIFF
    elif method == 1:
        method = cv2.TM_SQDIFF_NORMED
    dst = cv2.matchTemplate(image, templ, method)
    return dst


if __name__ == '__main__':
    find_object()


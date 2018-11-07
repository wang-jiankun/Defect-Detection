"""
图像分割，分割出铝型材的区域
author: 王建坤
date: 2018-9-25
"""
import cv2
import numpy as np
from PIL import Image, ImageFile
import time
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 600
IMG_SIZE1 = 300
METHOD = 0
SHOW = True


def data_image_segm(root_path, save_dir):
    """
    分割数据集中的一类图片
    :param root_path: 该类图片的根路径
    :param save_dir: 保存路径
    :return: none
    """
    img_list = os.listdir(root_path)
    for img_name in img_list:
        print(img_name)
        img_path = os.path.join(root_path, img_name)
        img = cv2.imread(img_path)
        # img = Image.open(img_path)
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dst = image_segm(img_gray)
        save_img_dir = os.path.join(save_dir, img_name)
        # dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        # dst.save(save_img_dir)
        cv2.imwrite(save_img_dir, dst)


def image_segm_single(img_path):
    """
    分割指定路径的单张图像
    :return: none
    """
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    image_segm(img)


def image_segm(img, method=METHOD):
    """
    分割 numpy array 的图像
    :param img: numpy array 的图像
    :param method:
    :return:
    """
    start_time = time.clock()
    # img_gray = cv2.split(img)[2]
    # img_blur = cv2.medianBlur(img, 5)
    # img_sobel = single_sobel(img_gray, 0, 1)
    img_grad = img_gradient(img)
    img_bin = img_thresh(img_grad, 100, 255)
    # img_mor = img_morphology(img_grad)
    if method == 0:
        points = hough_lines_prob(img_bin, img)
    else:
        points = find_contours(img_bin, img)
    dst = perspective_transform(img, points)
    end_time = time.clock()
    print('run time: ', end_time-start_time)

    if SHOW:
        cv2.imshow('img_grad', img_grad)
        cv2.imshow('img_bin', img_bin)
        cv2.imshow('img_src', img)
        cv2.imshow('dst', dst)

        cv2.waitKey()
        cv2.destroyAllWindows()
    return dst


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
    图像的梯度图
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
    :param img_gray:
    :param thresh:
    :param max_val:
    :return:
    """
    # 普通阈值：cv2.THRESH_BINARY,自适应阈值：cv2.THRESH_OTSU, cv2.THRESH_TRUNC
    _, dst = cv2.threshold(img_gray, thresh, max_val, cv2.THRESH_BINARY)
    return dst


def img_morphology(img_gray):
    """
    形态学操作
    :param img_gray:
    :return:
    """
    # 腐蚀
    # img_gray = cv2.erode(img_gray, (5, 5), iterations=2)
    # 膨胀
    img_gray = cv2.dilate(img_gray, (3, 3), iterations=2)
    return img_gray


def canny_detect(img_gray):
    """
    canny边缘检测
    :param img_gray:
    :return:
    """
    dst = cv2.Canny(img_gray, 30, 200)
    return dst


def hough_lines(edges):
    """
    标准霍夫线检测
    :param edges:
    :return:
    """
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    return lines


def hough_lines_prob(edges, img, draw=True):
    """
    概率霍夫线检测
    :param edges:
    :param img:
    :param draw:
    :return:
    """
    min_line_length = IMG_SIZE/2.5
    max_line_gap = max(IMG_SIZE/120, 2)
    # 检测结果为表示N条直线的 Nx1x4 的数组
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, None, min_line_length, max_line_gap)
    points = filter_lines(img, lines)
    if draw:
        cv2.polylines(img, np.array([[points[0], points[1], points[3], points[2]]]), True, (0, 255, 255), 2)
    return points


def draw_lines(img, lines):
    """
    画多条线
    :param img:
    :param lines:
    :return:
    """
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        print(x1, y1, x2, y2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def filter_lines(img, lines):
    """
    过滤经过霍夫线检测得到的线,得到目标的边界线
    :param img:
    :param lines:
    :return:
    """
    if lines is None:
        return
    lines_kb, border_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k = (y2 - y1) / (x2 - x1 + 0.0001)
        b = y1 - k * x1
        flag = True
        # 如果和已有的直线离得太近，则当前这条直线不添加
        # for bb in lines_kb:
        #     if abs(b - bb)/abs(b) < 0.2:
        #         flag = False
        #         break
        # 计算直线的两点坐标
        if flag:
            # x_1 = 0
            # x_2 = IMG_SIZE
            # y_1 = int(b + 0.5)
            # y_2 = int(k * IMG_SIZE + b + 0.5)
            border_lines.append(line)  # [[x_1, y_1, x_2, y_2]]
            lines_kb.append([k, b])

    if len(lines_kb) < 4:
        print('检测到：', len(lines_kb), '条直线')
        draw_lines(img, border_lines)
        return

    return four_lines(lines_kb)


def two_lines(lines_b, border_lines):
    """
    保留两条线，坐标为与边界的交点
    :param lines_b:
    :param border_lines:
    :return:
    """
    if len(lines_b) < 2:
        return
    index = np.argsort(lines_b)
    if abs(lines_b[index[0]] - lines_b[index[-1]]) < IMG_SIZE/5:
        return
    lines = [border_lines[index[0]], border_lines[index[-1]]]
    points = [[lines[0][0][0], lines[0][0][1]], [lines[0][0][2], lines[0][0][3]],
              [lines[1][0][0], lines[1][0][1]], [lines[1][0][2], lines[1][0][3]]]
    return points


def four_lines(lines_kb):
    """
    保留四条线，坐标为四条线的四个交点
    :param lines_kb:
    :return:
    """
    points = []
    m_index = 0
    lines_kb.sort(key=lambda e: e[1])
    for i in range(len(lines_kb)):
        if lines_kb[i][0] < 0:
            m_index = i
            break
    four_kb = [lines_kb[0], lines_kb[m_index-1], lines_kb[m_index], lines_kb[-1]]

    for i in range(2):
        for j in range(2, 4):
            x = (four_kb[j][1]-four_kb[i][1])/(four_kb[i][0]-four_kb[j][0]+0.001)
            y = four_kb[i][0]*x + four_kb[i][1]
            points.append([int(x), int(y)])
    print(points)
    return points


def perspective_transform(img, points, method=METHOD):
    """
    投影变换
    :param img:
    :param points:
    :param method:
    :return:
    """
    if points is None or len(points) < 2:
        dst = cv2.resize(img, (IMG_SIZE1, IMG_SIZE1))
    else:
        pts1 = np.float32(points)
        if method == 0:
            pts2 = np.float32([[0, 0], [IMG_SIZE1, 0], [0, IMG_SIZE1], [IMG_SIZE1, IMG_SIZE1]])
        else:
            pts2 = np.float32([[0, IMG_SIZE1], [0, 0], [IMG_SIZE1, 0], [IMG_SIZE1, IMG_SIZE1]])
        transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, transform_matrix, (IMG_SIZE1, IMG_SIZE1))
    return dst


def find_contours(img_bin, img, draw=True):
    """
    查找轮廓
    :param img_bin:
    :param img:
    :param draw:
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
        for contour in contours:
            area = cv2.contourArea(contour)
            if 360000/5 < area:
                break
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        print(box)
        if box[0][0] > IMG_SIZE/2:
            box = box[[1, 2, 3, 0], :]
        if draw:
            cv2.polylines(img, [box], True, (0, 0, 255), 3)
    elif approximate_method == 2:
        box = []
    else:
        if draw:
            cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        box = []
    return box


def draw_contours(img, lines):
    """
    画多边形
    :param img:
    :param lines:
    :return:
    """
    contours = np.array([[[[lines[0][0][0], lines[0][0][1]]], [[lines[0][0][2], lines[0][0][3]]],
                          [[lines[1][0][2], lines[1][0][3]]], [[lines[1][0][0], lines[1][0][1]]]]])
    cv2.drawContours(img, contours, -1, (255, 0, 0), thickness=2)


def fill_rev_roi(img, lines):
    """
    填充感兴趣区域以外的区域
    :param img:
    :param lines:
    :return:
    """
    dots = np.array([[[IMG_SIZE, 0], [0, 0], [lines[0][0][0], lines[0][0][1]], [lines[0][0][2], lines[0][0][3]]],
                     [[0, IMG_SIZE], [IMG_SIZE, IMG_SIZE], [lines[1][0][2], lines[1][0][3]],
                      [lines[1][0][0], lines[1][0][1]]]], dtype=np.int32)
    cv2.fillPoly(img, dots, (0, 0, 0))


def flood_fill(img):
    """
    漫水填充
    :param img:
    :return:
    """
    mask = np.zeros((IMG_SIZE + 2, IMG_SIZE + 2), np.uint8)
    cv2.floodFill(img, mask, (300, 300), 255, (20, 20, 20), (20, 20, 20))


def water_shed(img):
    """
    分水岭，没调试成功
    :param img:
    :return:
    """
    markers = np.zeros((IMG_SIZE, IMG_SIZE), np.int32)
    markers[::30, ::30] = 200
    cv2.watershed(img, markers)
    print(markers)
    out = cv2.convertScaleAbs(markers)
    cv2.imshow('water_shed', out)


if __name__ == '__main__':
    print('running image segmentation:')
    # data_image_segm('../data/phone', '../data/seg/phone')
    image_segm_single('../data/phone/10_23(14_19_11_703).jpg')  # 14_19_23_581

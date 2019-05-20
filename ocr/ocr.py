"""
电子烟雾化器产品批号字符识别
@Author: JK_Wang
@Time: 13-May-19
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROI = {'x1': 130, 'y1': 90, 'x2': 440, 'y2': 195}
THRESH = 80


def c_roi(img_path):
    img = cv2.imread(img_path, 0)
    roi_img = img[ROI['y1']:ROI['y2'], ROI['x1']:ROI['x2']]
    cv2.imshow('1', roi_img)
    return roi_img


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


def c_location(roi_img):
    # blur_img = cv2.medianBlur(roi_img, 3)
    # blur_img = cv2.equalizeHist(blur_img)
    blur_img = cv2.bilateralFilter(roi_img, 9, 5, 5)
    cv2.imshow('2', blur_img)

    edge = cv2.Canny(blur_img, 50, 150)
    cv2.imshow('3', edge)

    _, th_img = cv2.threshold(roi_img, THRESH, 255, cv2.THRESH_OTSU)  # cv2.THRESH_BINARY
    cv2.imshow('4', th_img)

    # loc_img = cv2.addWeighted(edge, 1, th_img, 1, 0)
    loc_img = img_morphology(th_img, 2, 5)
    loc_img = img_morphology(loc_img, 3, 9)
    # loc_img = th_img

    v_sum = np.sum(loc_img, 0, dtype=np.float)
    seg_pos = []
    x, y = 0, 0
    for i in range(len(v_sum)-1):
        if v_sum[i] == 0 and v_sum[i+1] != 0:
            x = i
        elif v_sum[i] != 0 and v_sum[i+1] == 0:
            y = i
            seg_pos.append([x, y])

    k = 10
    for i, j in seg_pos:
        c_img = roi_img[:, i:j]
        c_img = cv2.equalizeHist(c_img)
        _, c_img = cv2.threshold(c_img, THRESH, 255, cv2.THRESH_OTSU)
        c_img = img_morphology(c_img, 3, 3)
        k += 1
        cv2.imshow(str(k), c_img)

    plt.plot(v_sum)
    # plt.axis([0, 250, 0, 140])
    plt.xlabel('列坐标', fontproperties='SimHei', fontsize=14)
    plt.ylabel('白点数量', fontproperties='SimHei', fontsize=14)
    plt.show()

    return loc_img


def ocr(img_path):
    roi_img = c_roi(img_path)
    loc_img = c_location(roi_img)
    cv2.imshow('img', loc_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ocr('E:/Defect_Detection/data/ocr/2.png')


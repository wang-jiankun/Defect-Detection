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
import matplotlib


# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun'

# matplotlib.rcParams['font.family'] = 'SimHei'
# matplotlib.rcParams['font.size'] = 20
IMG_H_SIZE = 2000
IMG_V_SIZE = 1200
TEMPL_DIR = '../data/cigarette/template/'
SIZE_RATIO = 4
SAVE_PATH = 'E:/test/'
DETECT_CLASS = 4
# 边框的颜色和线宽
COLOR_1 = (255, 255, 255)
LINE_1 = 5
COLOR_2 = (0, 0, 255)
LINE_2 = 5


class AssembleDetection:
    """
    装配正确性检测类
    """
    def __init__(self, img_path):
        self.gray_img = self.open_img(img_path, resize=True)
        self.color_img = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2BGR)
        self.roi = {}
        self.res = [0]*5

    @staticmethod
    def open_img(img_path, resize=True):
        """
        打开指定路径的单张图像（灰度图），并缩放
        :param img_path: 图片路径
        :param resize: 是否要缩放图片
        :return:
        """
        img = cv2.imread(img_path, 0)
        if resize:
            img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        return img

    @staticmethod
    def show_save_img(img, name):
        """
        显示并保存图片
        :param img:
        :param name:
        :return:
        """
        cv2.imshow(name, img)
        cv2.imwrite(os.path.join(SAVE_PATH, name+'.jpg'), img)

    def find_position(self, thresh=61):
        """
        找到检测目标的位置
        :param thresh: 阈值
        :return:
        """
        #  二值化
        thresh_img = self.img_thresh(self.gray_img, thresh, 255, 1)

        # self.show_save_img(thresh_img, 'object_thresh')
        # 开运算
        open_img = self.img_morphology(thresh_img, 2, 15)

        open_img = self.img_morphology(open_img, 3, 9)

        # 找到工件的位置
        box = self.find_object(open_img, self.color_img, 0, True)
        return box

    @staticmethod
    def img_thresh(img_gray, thresh, max_val, method=0):
        """
        图像二值化
        :param img_gray: 灰度图
        :param thresh: 阈值
        :param max_val: 二值中的高值
        :param method: 方法
        :return:
        """
        # 自适应阈值
        # dst = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
        # 普通阈值：cv2.THRESH_BINARY,最佳阈值：cv2.THRESH_OTSU+cv2.THRESH_BINARY,
        if method == 2:
            thresh = int(np.max(img_gray[0:100, 220]))
        _, dst = cv2.threshold(img_gray, thresh, max_val, cv2.THRESH_BINARY)

        return dst

    @staticmethod
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

    @staticmethod
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
        # print('contours:', len(contours))
        temp = np.zeros(img.shape)
        box = None
        # 轮廓是近似方式，0：矩形，1：最小矩形，2：点集
        if approximate_method == 0:
            area_thresh = img.shape[0] * img.shape[1] / SIZE_RATIO
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > area_thresh:
                    box = x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_1, LINE_1)
                    # cv2.imshow('temp',img)
                    # cv2.imwrite(os.path.join(SAVE_PATH, 'contours.jpg'), img)
                    # cv2.waitKey()

        elif approximate_method == 1:
            contour = 0
            area_thresh = img.shape[0] * img.shape[1] / SIZE_RATIO
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > area_thresh:
                    break
            rect = cv2.minAreaRect(contour)
            box = np.int0(cv2.boxPoints(rect))
            if draw:
                cv2.polylines(img, [box], False, 255, 1)
        else:
            if draw:
                cv2.drawContours(temp, contours, -1, COLOR_1, LINE_1)
                # cv2.imshow('temp', temp)
                # cv2.imwrite(os.path.join(SAVE_PATH, 'contours.jpg'), temp)
                # cv2.waitKey()
                # temp = np.zeros(img.shape)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    area_thresh = img.shape[0] * img.shape[1] / SIZE_RATIO
                    if area > 100:
                        box = x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(temp, (x, y), (x + w, y + h), COLOR_1, LINE_1)
                        cv2.imshow('temp', temp)
                        cv2.imwrite(os.path.join(SAVE_PATH, 'x.jpg'), temp)
                        cv2.waitKey()

        return box

    def set_roi(self, box, show=True, draw=True):
        """
        设置需要检测的 ROI
        :param box:
        :param draw:
        :param show:
        :return:
        """
        x, y, w, h = box
        # 正中间
        # center_roi = {'x1': int(x+2/5*w), 'y1': int(y+2/5*h), 'x2': int(x+3/5*w), 'y2': int(y+3/5*h)}
        # center_img = self.intercept_roi(center_roi)
        # self.roi['c'] = center_img

        # 左中
        lc_roi = {'x1': int(x+1/7*w), 'y1': int(y+3/7*h), 'x2': int(x+3/14*w), 'y2': int(y+4/7*h)}
        lc_img = self.intercept_roi(lc_roi)
        self.roi['lc'] = lc_img

        # 右中
        rc_roi = {'x1': int(x+11/14*w), 'y1': int(y+3/7*h), 'x2': int(x+6/7*w), 'y2': int(y+4/7*h)}
        rc_img = self.intercept_roi(rc_roi)
        self.roi['rc'] = rc_img

        # 顶部
        top_roi = {'x1': int(x+w*3/5), 'y1': max(0, int(y-h/6)), 'x2': int(x+w*7/10), 'y2': int(y-10)}
        top_img = self.intercept_roi(top_roi)
        self.roi['top'] = top_img

        # 底部
        bottom_roi = {'x1': int(x+w*3/10), 'y1': int(y+h+10), 'x2': int(x+w*2/5), 'y2': min(1000, int(y+h*7/6))}
        bottom_img = self.intercept_roi(bottom_roi)
        self.roi['bottom'] = bottom_img

        # 左上
        lt_roi = {'x1': int(x+w*1/4), 'y1': int(y+h*3/20), 'x2': int(x+w*2/5), 'y2': int(y+h*21/60)}
        lt_img = self.intercept_roi(lt_roi)
        self.roi['lt'] = lt_img

        # 左下
        lb_roi = {'x1': int(x+w*1/4), 'y1': int(y+h*2/3), 'x2': int(x+w*2/5), 'y2': int(y+h*17/20)}
        lb_img = self.intercept_roi(lb_roi)
        self.roi['lb'] = lb_img

        # 右上
        rt_roi = {'x1': int(x+w*3/5), 'y1': int(y+h*3/20), 'x2': int(x+w*3/4), 'y2': int(y+h*1/3)}
        rt_img = self.intercept_roi(rt_roi)
        self.roi['rt'] = rt_img

        # 右下
        rb_roi = {'x1': int(x+w*3/5), 'y1': int(y+h*39/60), 'x2': int(x+w*3/4), 'y2': int(y+h*17/20)}
        rb_img = self.intercept_roi(rb_roi)
        self.roi['rb'] = rb_img

        if draw:
            # self.draw_roi(center_roi)
            self.draw_roi(lc_roi)
            self.draw_roi(rc_roi)
            self.draw_roi(top_roi)
            self.draw_roi(bottom_roi)
            self.draw_roi(lt_roi)
            self.draw_roi(lb_roi)
            self.draw_roi(rt_roi)
            self.draw_roi(rb_roi)

        if show:
            # cv2.imshow('center', center_img)
            cv2.imshow('lc', lc_img)
            cv2.imshow('rc', rc_img)
            cv2.imshow('top', top_img)
            cv2.imshow('bottom', bottom_img)
            cv2.imshow('lt', lt_img)
            cv2.imshow('lb', lb_img)
            cv2.imshow('rt', rt_img)
            cv2.imshow('rb', rb_img)

    def intercept_roi(self, roi_position):
        """
        截取图片的指定 ROI
        :param roi_position:
        :return:
        """
        return self.gray_img[roi_position['y1']:roi_position['y2'], roi_position['x1']:roi_position['x2']]

    def draw_roi(self, roi):
        """
        绘制 ROI 框
        :param roi:
        :return:
        """
        cv2.rectangle(self.color_img, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), COLOR_2, LINE_2)

    def roi_pre(self, img):
        """
        ROI 图像预处理
        :param img:
        :return:
        """
        # 闭运算
        temp_img = self.img_morphology(img, 3, 9)
        return temp_img

    @staticmethod
    def template_match(image, templ, method=0):
        """
        进行模板匹配
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

    @staticmethod
    def find_extreme_index(array):
        """
        找极大值的下标
        :param array: 数组
        :return:
        """
        index_list = []
        for i in range(array.shape[0]-1):
            if array[i+1] > array[i] and array[i+1] >= array[i+2]:
                index_list.append(i)
        return index_list

    def object_detect(self, box):
        """
        工件检测
        :param box:
        :return:
        """
        if box:
            print('Detect object:', 'OK', box)
            self.res[0] = 0
        else:
            print('Detect object:', 'NG')
            self.res[0] = 1

    def cotton_detect(self, thresh=230): #230
        """
        棉芯检测
        :param thresh:
        :return:
        """
        lc_img = self.roi['lc']
        rc_img = self.roi['rc']
        lc_img = self.img_thresh(lc_img, thresh, 1)
        rc_img = self.img_thresh(rc_img, thresh, 1)
        area = lc_img.shape[0]*lc_img.shape[1]
        lc_sum = np.sum(lc_img)
        rc_sum = np.sum(rc_img)
        if lc_sum/area < 0.9:
            print('Cotton detect: NG ', 'lc')
            self.res[1] = 1
        elif rc_sum/area < 0.9:
            print('Cotton detect: NG ', 'rc')
            self.res[1] = 1
        else:
            print('Cotton detect: OK')
            self.res[1] = 0

    def limit_detect(self, thresh=200):
        """
        金属丝的限位检测
        :param thresh:
        :return:
        """
        top_img = self.roi['top']
        bottom_img = self.roi['bottom']
        top_img = self.img_thresh(top_img, thresh, 1)
        bottom_img = self.img_thresh(bottom_img, thresh, 1)
        lc_sum = np.sum(top_img)
        rc_sum = np.sum(bottom_img)
        if lc_sum < 10:
            print('Limit detect: NG ', 'top')
            self.res[2] = 1
        elif not rc_sum or rc_sum < 10:
            print('Limit detect: NG ', 'bottom')
            self.res[2] = 1
        else:
            print('Limit detect: OK')
            self.res[2] += 0

    def piece_detect(self, roi_name, templ_img, thresh=220):
        """
        金属片检测
        :param roi_name:
        :param templ_img:
        :param thresh:
        :return:
        """
        roi_img = self.roi[roi_name]
        roi_img = self.roi_pre(roi_img)
        roi_img = self.img_thresh(roi_img, thresh, 1)
        templ_img = self.img_thresh(templ_img, thresh, 1)
        dst = self.template_match(roi_img, templ_img, 1)
        min_value = np.min(dst)
        if min_value < 0.4:
            print(roi_name, 'Piece detect:', 'OK', 'match -', min_value)
            self.res[3] += 0
        else:
            print(roi_name, 'Piece detect:', 'NG', 'match -', min_value)
            self.res[3] = 1

    def wire_detect(self, roi_name, thresh=240):
        """
        金属丝安装位置检测
        :param roi_name:
        :param thresh:
        :return:
        """
        roi_img = self.roi[roi_name]
        roi_img = self.roi_pre(roi_img)
        roi_img = self.img_thresh(roi_img, thresh, 1, 1)
        # 列白色像素统计
        v_sum = np.sum(roi_img, 0, dtype=np.float)
        # 数值过滤
        v_sum[v_sum < roi_img.shape[0]/2] = 0.0
        # 平滑
        v_sum = np.convolve(v_sum, [0.1, 0.2, 0.4, 0.2, 0.1])
        nonzero_index = v_sum.nonzero()

        try:
            dist = nonzero_index[0][-1] - nonzero_index[0][0]
            if 24 < dist < 38:
                print(roi_name, 'Wire detect: ', 'OK', 'dist -', dist)
                self.res[4] += 0
            else:
                print(roi_name, 'Wire detect: ', 'NG', 'dist -', dist)
                self.res[4] = 1
        except:
            self.res[4] = 1

    def detect(self):
        """
        装配正确性检测
        :return:
        """
        # 工件检测
        box = self.find_position()
        self.object_detect(box)
        cv2.waitKey()
        if self.res[0]:
            pre = 1
            return pre
        # 设置 ROI
        self.set_roi(box, False)
        # 棉芯检测
        self.cotton_detect()
        # 金属线检测
        self.limit_detect()
        # 左上 ROI 检测
        templ_img1 = self.open_img(TEMPL_DIR + 'lt.jpg')
        self.piece_detect('lt', templ_img1)
        # 右下 ROI 检测
        templ_img2 = self.open_img(TEMPL_DIR + 'rb.jpg')
        self.piece_detect('rb', templ_img2)
        # 左下 ROI 检测
        self.wire_detect('lb')
        # 右上 ROI 检测
        self.wire_detect('rt')

        pre = 0
        for i in range(5):
            if self.res[i] != 0:
                if i == 1:
                    pre = 2
                elif i == 3:
                    pre = 3
                else:
                    pre = 4
                break

        # 显示图片
        return pre


def single_detect(img_path):
    """
    检测单张图片
    :return:
    """
    print('Detecting image:', img_path)
    start_time = time.clock()
    detector = AssembleDetection(img_path)
    p = detector.detect()
    end_time = time.clock()
    run_time = round((end_time - start_time)*1000, 1)
    return p, run_time


if __name__ == '__main__':
    # normal, nothing, lack_cotton, lack_piece, wire_fail
    single_detect('E:/test/0.jpg')
    # folder = '../data/backup/cigarette/wire_fail'
    # folder = '../data/cigarette/normal/'
    # folder_detect(folder)
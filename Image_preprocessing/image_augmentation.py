"""
图像数据扩增
author: 王建坤
date: 2018-9-29
"""
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image

LABEL_LIST = []


def create_image_list(image_dir):
    """
    生成一个以类别文件名为 key, 图片列表为 value 的字典
    :param image_dir:
    :return
    """
    class_dic = {}
    for root, dirs, files in os.walk(image_dir):
        for label in LABEL_LIST:
            if root.split('\\')[-1] == label:
                tem_list = []
                for f in files:
                    if f.split('.')[-1] == 'jpg':
                        path = os.path.join(root, f)
                        tem_list.append(path)
                class_dic[label] = tem_list
    return class_dic


def five_crop(image, crop_ratio):
    """
    裁剪图片的左上、左下、右上、右下、中间子块作为训练样本，如图片尺寸为[224,224]，裁剪左上角大小为[196,196]的子块可增加训练模型鲁棒性
    """
    original_image = image
    row = original_image.shape[0]
    col = original_image.shape[1]
    image_size = (int(row * crop_ratio), int(col * crop_ratio))
    image_left_top = original_image[:image_size[0], :image_size[1]]
    image_right_top = original_image[:image_size[0], col - image_size[1]:]
    image_left_bottom = original_image[row - image_size[0]:, :image_size[1]]
    image_right_bottom = original_image[row - image_size[0]:, col - image_size[1]:]
    row_start = int((row - image_size[0]) / 2)
    col_start = int((col - image_size[1]) / 2)
    image_center = original_image[row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
    image_list = [image_left_top, image_right_top, image_left_bottom, image_right_bottom, image_center]
    image_list = [cv2.resize(img, (col, row)) for img in image_list]
    return image_list


def random_crop(image, crop_ratio, num):
    """
    随机裁剪图片
    """
    original_image = image
    row = original_image.shape[0]
    col = original_image.shape[1]
    image_size = (int(row * crop_ratio), int(col * crop_ratio))
    image_left_top = original_image[:image_size[0], :image_size[1]]
    image_right_top = original_image[:image_size[0], col - image_size[1]:]
    image_left_bottom = original_image[row - image_size[0]:, :image_size[1]]
    image_right_bottom = original_image[row - image_size[0]:, col - image_size[1]:]
    row_start = int((row - image_size[0]) / 2)
    col_start = int((col - image_size[1]) / 2)
    image_center = original_image[row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
    crop_list = [image_left_top, image_right_top, image_left_bottom, image_right_bottom, image_center]
    crop_list = [cv2.resize(img, (col, row)) for img in crop_list]
    image_list = []
    state = random.randint(0, 4)
    for _ in range(num):
        image = crop_list[state]
        image_list.append(image)
    return image_list


def horizontal_flip(image):
    """
    对图片进行水平翻转
    """
    original_image = image
    flip_h = cv2.flip(original_image, 1)
    return flip_h


def vertical_flip(image):
    """
    对图片进行垂直翻转
    """
    original_image = image
    flip_v = cv2.flip(original_image, 0)
    return flip_v


def rotate(image):
    """
    对图片进行小角度旋转
    """
    original_image = image
    row = original_image.shape[0]
    col = original_image.shape[1]
    row_center = int(row / 2)
    col_center = int(col / 2)
    image_list = []
    rotate_degree_list = [5, 10, 15, 20]
    for rotate_degree in rotate_degree_list:
        rotate_matrix = cv2.getRotationMatrix2D((col_center, row_center), rotate_degree, 1)
        rotate_image = cv2.warpAffine(original_image, rotate_matrix, (col, row))
        image_list.append(rotate_image)
    else:
        return image_list


def rotate_enhance(image_path_list, save_path):
    """
    对图片进行小角度旋转,并保存
    :param image_path_list: 图片路径列表
    :param save_path: 保存路径
    :return: none
    """
    print('rotate augmentation:')
    for image_path in image_path_list:
        # original_image = cv2.imread(image_path)
        original_image = Image.open(image_path)
        original_image = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR)
        row = original_image.shape[0]
        col = original_image.shape[1]
        row_center = int(row / 2)
        col_center = int(col / 2)
        rotate_degree_list = [-5, 5]
        for rotate_degree in rotate_degree_list:
            rotate_matrix = cv2.getRotationMatrix2D((col_center, row_center), rotate_degree, 1.1)
            rotate_image = cv2.warpAffine(original_image, rotate_matrix, (col, row))
            img_pil = Image.fromarray(cv2.cvtColor(rotate_image, cv2.COLOR_BGR2RGB))
            prefix = image_path.split('\\')[-1].split('.')[0]
            img_save_path = save_path + '/' + prefix + '_r%d.jpg' % rotate_degree
            img_pil.save(img_save_path)


def geometry_enhance(image_path_list, save_path):
    """
    几何增强，根据增强的类型添加的图片后缀名
    :param image_path_list:
    :param save_path:
    :return:
    """
    name_list = ['hf', 'r1', 'r2', 'r3', 'r4', 'lt', 'rt', 'lb', 'rb', 'c']
    for path in image_path_list:
        original_image = cv2.imread(path, 1)
        image_list = []
        flip = horizontal_flip(original_image)
        rotate_list = rotate(original_image)
        crop_list = five_crop(original_image, 0.8)
        image_list.append(flip)
        image_list.extend(rotate_list)
        image_list.extend(crop_list)
        prefix = path.split('\\')[-1]
        prefix = prefix.split('.')[0]
        prefix = save_path + '/' + prefix
        for image, name in zip(image_list, name_list):
            cv2.imwrite(prefix + '_%s.jpg' % name, image)


def color_enhance(image_path_list, save_path):
    """
    颜色增强，对图片进行色彩抖动，改变亮度、对比度、锐度（ImageEnhance.Sharpness）
    :param image_path_list:
    :param save_path:
    :return:
    """
    print('color augmentation:')
    for path in image_path_list:
        original_image = Image.open(path)
        for i in range(2):
            # 色彩抖动
            color_factor = np.random.randint(9, 11) / 10.
            color_image = ImageEnhance.Color(original_image).enhance(color_factor)
            # 亮度
            brightness_factor = np.random.randint(9, 11) / 10.
            brightness_image = ImageEnhance.Brightness(color_image).enhance(brightness_factor)
            # 对比度
            contrast_factor = np.random.randint(9, 11) / 10.
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(contrast_factor)
            prefix = path.split('\\')[-1]
            prefix = prefix.split('.')[0]
            prefix = save_path + '/' + prefix + '_c%d.jpg' % (i+1)
            contrast_image.save(prefix)


def alum_enhance():
    """
    对铝型材数据集进行数据增强
    :return:
    """
    class_root_path = 'E:/dataset/alum/guangdong_round1_train2_20180916/瑕疵样本'
    save_root_path = 'E:/dataset/alum/guangdong_round1_train2_20180916/瑕疵样本'
    class_list = os.listdir(class_root_path)
    for image_root_path in class_list:
        if image_root_path == '其他':
            continue
        print('The class in processing is', image_root_path)
        save_path = save_root_path + '/' + image_root_path
        image_root_path = class_root_path + '/' + image_root_path
        image_name_list = os.listdir(image_root_path)
        if len(image_name_list) > 200:
            continue
        image_path_list = []
        for image_name in image_name_list:
            image_path = os.path.join(image_root_path, image_name)
            image_path_list.append(image_path)
        rotate_enhance(image_path_list, save_path)
        # color_enhance(image_path_list, save_path)


def gray_enhance(image_path_list, save_path, num=2):
    """
    灰度增强，对图片进行亮度、对比度变换增强
    :param image_path_list:
    :param save_path:
    :param num:
    :return:
    """
    print('gray augmentation:')
    for path in image_path_list:
        print(path)
        original_image = Image.open(path)
        for i in range(num):
            # 亮度
            # brightness_factor = 1.0
            brightness_factor = np.random.randint(98, 106) / 100.
            brightness_image = ImageEnhance.Brightness(original_image).enhance(brightness_factor)
            # 对比度
            # contrast_factor = 1.5
            contrast_factor = np.random.randint(95, 105) / 100.
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(contrast_factor)
            prefix = path.split('/')[-1]
            prefix = prefix.split('.')[0]
            save_name = save_path + '/' + prefix + '_g%d.jpg' % (i+1)
            contrast_image.save(save_name)

            # 显示图片
            # brightness_image.show()
            # brightness_image.save('../data/cigarette/aug/bright.jpg')
            # cv2.waitKey()
            # cv2.destroyAllWindows()


def translation_enhance(image_path_list, save_path, num=2):
    print('translation augmentation:')
    for path in image_path_list:
        print(path)
        original_image = cv2.imread(path)
        for i in range(num):
            tx = np.random.randint(10, 50)
            ty = np.random.randint(10, 30)
            # tx, ty = 50, 60
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            rows, cols = original_image.shape[0], original_image.shape[1]
            trans_img = cv2.warpAffine(original_image, M, (cols, rows))
            prefix = path.split('/')[-1]
            prefix = prefix.split('.')[0]
            save_name = save_path + '/' + prefix + '_t%d.jpg' % (i+1)
            cv2.imwrite(save_name, trans_img)

            # cv2.imshow('trans', trans_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()


def cig_enhance():
    """
    对电子烟数据集进行数据增强
    :return:
    """
    data_root_path = '../data/cigarette'
    save_path = False
    class_list = ['normal', 'nothing', 'lack_cotton', 'lack_piece', 'wire_fail', 'aug']
    folder_list = os.listdir(data_root_path)
    for folder in ['aug']:
        if folder not in class_list:
            continue
        print('The folder in processing is', folder)
        image_root_path = data_root_path + '/' + folder
        if save_path:
            save_path = '../data/cigarette/aug'
        else:
            save_path = image_root_path
        image_name_list = os.listdir(image_root_path)
        image_path_list = []
        for image_name in image_name_list:
            image_path = image_root_path + '/' + image_name
            image_path_list.append(image_path)
        gray_enhance(image_path_list, save_path)
        translation_enhance(image_path_list, save_path)


if __name__ == '__main__':
    print('running data augmentation')
    cig_enhance()




"""
缺陷检测QT软件--图像处理
author: 王建坤
date: 2018-10-17
"""
import cv2


def put_text(img_path, pre_class):
    """
    打开图片绘制文字
    :param img_path:
    :param pre_class:
    :return:
    """
    # class_name_dic = {0: '正常', 1: '不导电', 2: '擦花', 3: '角位漏底', 4: '桔皮', 5: '漏底', 6: '起坑', 7: '脏点'}
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500, 500))
    cv2.putText(img, str(pre_class), (img.shape[1]-100, img.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('1', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    put_text('../data/1.jpg', 1)
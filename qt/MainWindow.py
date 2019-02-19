"""
缺陷检测QT软件--主窗口类
author: 王建坤
date: 2018-9-25
"""
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QMessageBox, QTableWidgetItem, QHeaderView
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QPen, QBrush, QColor
from PyQt5.QtCore import Qt
from qt import predict_dl, LogDialog
import pandas as pd
import threading as th
from urllib import request
import sys
import os
import time
import logging
import zipfile
import pymysql

__version__ = '1.0'


class Detect(QMainWindow):
    def __init__(self, *args):
        super(Detect, self).__init__(*args)
        # 在窗口中加载界面
        loadUi('ui_detection.ui', self)
        self.image = None
        self.image_list = []
        self.image_index = 0
        self.folder_path = None

        # 连接数据库
        self.connection = pymysql.connect(host='localhost', user='root', password='1234',
                                          db='detection_data', charset='utf8')
        self.cursor = self.connection.cursor()
        # self.class_name_dic = {0: '正常', 1: '不导电', 2: '划痕', 3: '污渍', 4: '桔皮', 5: '漏底', 6: '起坑', 7: '脏点'}
        self.class_name_dic = {0: 'normal', 1: 'nothing', 2: 'lack_cotton', 3: 'lack_piece', 4: 'wire_fail'}
        # 主窗口，信号与槽绑定，初始化设置
        self.ac_exit.triggered.connect(QApplication.exit)
        self.ac_detect_log.triggered.connect(self.action_detect_log)
        self.ac_defect_log.triggered.connect(self.action_defect_log)
        self.ac_update_model.triggered.connect(self.action_update_model)
        self.ac_open_website.triggered.connect(self.action_open_website)
        self.pb_detect.clicked.connect(self.slot_detect)
        self.pb_choose_image.clicked.connect(self.slot_image_browser)
        self.pb_choose_folder.clicked.connect(self.slot_folder_browser)
        self.pb_open_file.clicked.connect(self.slot_open_image)
        self.pb_next_image.clicked.connect(self.slot_next_image)
        self.pb_load_model.clicked.connect(self.slot_load_model)
        self.pb_load_model.hide()
        self.pb_close_model.clicked.connect(self.slot_close_model)
        self.pb_close_model.hide()
        self.pb_save_image.clicked.connect(self.slot_save_image)
        self.pb_detect.setDisabled(True)
        self.pb_next_image.setDisabled(True)
        self.pb_save_image.setDisabled(True)
        self.pb_close_model.setDisabled(True)
        self.table_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_history.horizontalHeader().setStyleSheet("QHeaderView::section{background-color:lightgray;};")
        # self.table_history.horizontalHeaderItem(0).setBackground(QBrush(QColor(0,0,0))

        # print 输出重定向
        sys.stdout = EmittingStream(textWritten=self.my_output)
        sys.stderr = EmittingStream(textWritten=self.my_output)
        print('Welcome to use detection system.')

        # 初始化
        self.slot_load_model()
        print('Initialization complete!')
        self.statusBar().showMessage('   就绪')

    def slot_image_browser(self):
        """
        选择图片槽函数
        :return:
        """
        # 打开文件浏览器，获得选择的文件
        image_path = QFileDialog.getOpenFileName(self, '选择图片', '/Defect_Detection/data/cigarette',
                                                 'JPEG(*.jpg) ;; PNG(*.png)')
        # 判断是否选择了文件
        if image_path[0] != '':
            # 显示文件名
            self.le_file.setText(image_path[0])
            self.slot_open_image()
            self.slot_detect()

    def slot_folder_browser(self):
        """
        选择目录槽函数
        :return:
        """
        # 打开文件浏览器，获得选择的文件
        self.folder_path = QFileDialog.getExistingDirectory(self, '选择目录', '/Defect_Detection/data/cigarette')
        # 判断是否选择了文件
        if self.folder_path != '':
            # 显示文件名
            self.image_index = 0
            self.image_list = os.listdir(self.folder_path)
            self.le_file.setText(self.folder_path + '/' + self.image_list[0])
            self.slot_open_image()
            self.slot_detect()
            self.pb_next_image.setEnabled(True)

    def slot_open_image(self):
        """
        打开图片槽函数
        :return:
        """
        # 获取文件名
        file_name = self.le_file.text()
        if file_name == '':
            QMessageBox.information(self, 'Error', '请输入图片路径！  ')
            return
        self.image = QImage(file_name)
        self.image = self.image.scaled(self.lb_image.size(), Qt.KeepAspectRatio)
        self.lb_image.setPixmap(QPixmap.fromImage(self.image))
        self.pb_save_image.setEnabled(True)
        self.pb_detect.setEnabled(True)

    def slot_next_image(self):
        """
        检测下一张图片
        :return:
        """
        self.image_index += 1
        if self.image_index == len(self.image_list):
            QMessageBox.information(self, '提示', '没有图片了  ')
            self.image_index -= 1
            return
        self.le_file.setText(self.folder_path + '/' + self.image_list[self.image_index])
        self.slot_open_image()
        self.slot_detect()

    def slot_detect(self):
        """
        检测槽函数
        :return:
        """
        img_path = self.le_file.text()
        if img_path == '':
            QMessageBox.information(self, 'Error', '请选择文件')
            return
        if self.cb_method.currentText() == '深度学习':
            pre, run_time = predict_dl.predict(img_path)
        else:
            pre, run_time = 0, 1
        row_count = self.table_history.rowCount()
        self.table_history.insertRow(row_count)
        self.table_history.setItem(row_count, 0, QTableWidgetItem(str(row_count+1)))
        self.table_history.item(row_count, 0).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.table_history.setItem(row_count, 1, QTableWidgetItem(self.class_name_dic[pre]))
        # 数据库插入一条检测记录
        self.insert_log(pre, img_path)
        # temp_log = [img_path, str(pre), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]
        if pre == -1:
            return
        # 在界面的历史记录表中插入信息
        self.le_class.setText(str(pre))
        self.le_time.setText(str(run_time))
        self.put_text(pre)

    def slot_load_model(self):
        """
        加载模型槽函数
        :return:
        """
        # model = self.cb_model.currentText()
        flag = predict_dl.load_model()
        if flag == -1:
            predict_dl.close_sess()
            return
        self.pb_load_model.setDisabled(True)
        # self.pb_detect.setEnabled(True)
        self.pb_close_model.setEnabled(True)

    def slot_close_model(self):
        """
        关闭模型槽函数
        :return:
        """
        self.pb_close_model.setDisabled(True)
        predict_dl.close_sess()
        self.pb_detect.setDisabled(True)
        self.pb_load_model.setEnabled(True)
        self.le_image_size.setEnabled(True)

    def slot_save_image(self):
        """
        保存检测结果的图片
        :return:
        """
        file_path = QFileDialog.getSaveFileName(self, 'save image', 'log/untitled.jpg', 'JPEG(*.jpg) ;; PNG(*.png)')
        if file_path[0] != '':
            print(file_path)
            self.image.save(file_path[0])
            print('Save detected image successfully')

    @staticmethod
    def action_detect_log(self):
        """
        打开检测记录文件窗口
        :return:
        """
        detect_dialog = LogDialog.DetectLog()
        detect_dialog.show()
        detect_dialog.exec()

    @staticmethod
    def action_defect_log(self):
        """
        打开缺陷记录文件窗口
        :return:
        """
        defect_dialog = LogDialog.DefectLog()
        defect_dialog.show()
        defect_dialog.exec()

    def action_update_model(self):
        """
        从网络下载ckpt文件，更新模型
        :return:
        """
        th_txt = th.Thread(target=self.download_txt, name='down_zip')
        th_txt.start()

    @staticmethod
    def action_open_website():
        """
        打开默认浏览器访问网址
        :return:
        """
        import webbrowser
        webbrowser.open('http://www.baidu.com')

    def download_txt(self):
        """
        下载软件最新消息，判断是否需要更新
        :return:
        """
        # print('running thread: ', th.current_thread().name)
        print("Downloading zip ......")
        url = 'https://codeload.github.com/wang-jiankun/hello-world/zip/master'
        try:
            request.urlretrieve(url, '1.zip')
        except Exception as e:
            logging.exception(e)
            print('下载 zip 出错')
            return
        print("Download zip completely")
        f = zipfile.ZipFile('1.zip', 'r')
        for file in f.namelist():
            f.extract(file)
        f.close()
        os.remove('1.zip')
        with open('hello-world-master/1.txt', 'r', encoding='utf-8') as f:
            version = f.readline().split()[1]
            print('当前版本：', __version__, '最新版本：', version)
            if version == __version__:
                print('已是最新版本，不用更新')
            else:
                model_path = f.readline().split()[1]
                th_model = th.Thread(target=self.download_model, name='down_model', args=(model_path,))
                th_model.start()

    @staticmethod
    def download_model(model_path):
        """
        下载软件的最新模型
        :return:
        """
        # print('running thread: ', th.current_thread().name)
        print("Downloading model ......")
        url = model_path
        try:
            request.urlretrieve(url, 'test1.jpg')
        except Exception as e:
            logging.exception(e)
            print('下载 model 出错')
            return
        print("Download model completely")
        print('Update model successfully')

    def closeEvent(self, event):
        """
        重写closeEvent方法，实现窗体关闭时执行其它操作
        :param event: close()触发的事件
        :return: None
        """
        reply = QMessageBox.question(self, '退出', '保存检测记录？',
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            csv = pd.DataFrame({'图片路径': [x[0] for x in self.log], '检测结果': [x[1] for x in self.log],
                                '检测时间': [x[2] for x in self.log]})
            csv.to_csv('log/detect_' + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.csv')
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

    def my_output(self, text):
        """
        输出重定向到 text edit 中
        :param text:
        :return:
        """
        cursor = self.te_log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        # if text == '\n':
        #     return
        # date = time.strftime('%Y-%m-%d %H:%M:%S : ', time.localtime())
        # cursor.insertText(date)
        cursor.insertText(text)
        # self.te_log.setTextCursor(cursor)
        # self.te_log.ensureCursorVisible()

    def refresh_para(self):
        """
        更新参数
        :return:
        """
        predict_dl.IMG_SIZE = int(self.le_image_size.text())
        # print('refresh: ', predict.IMG_SIZE)

    def put_text(self, pre):
        """
        图片添加检测结果信息
        :param pre:
        :return:
        """
        qp = QPainter()
        qp.begin(self.image)
        pen = QPen(Qt.yellow, 10, Qt.SolidLine)
        qp.setPen(pen)
        qp.setFont(QFont('Microsoft YaHei', 30))
        qp.drawText(550, 50, self.class_name_dic[pre])
        qp.end()
        self.lb_image.setPixmap(QPixmap.fromImage(self.image))

    def insert_log(self, pre, img_path):
        """
        向数据库中插入一条检测记录
        :param pre:
        :param img_path:
        :return:
        """
        # 插入一行
        sql = "INSERT INTO detection_log(detect_class, path) VALUES(%s, %s)"
        self.cursor.execute(sql, (self.class_name_dic[pre], img_path))
        # 提交事务
        self.connection.commit()


class EmittingStream(QtCore.QObject):
    """
    输出的自定义信号类
    """
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

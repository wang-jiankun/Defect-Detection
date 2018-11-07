"""
缺陷检测QT软件--主窗口类
author: 王建坤
date: 2018-9-25
"""
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QMessageBox, QTableWidgetItem
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QPen, QBrush, QColor
from PyQt5.QtCore import Qt
from qt import predict, LogDialog
import pandas as pd
import threading as th
from urllib import request
import sys
import os
import time
import logging
import zipfile

__version__ = '1.0'


class Detect(QMainWindow):
    def __init__(self, *args):
        super(Detect, self).__init__(*args)
        # 在窗口中加载界面
        loadUi('ui_detection.ui', self)
        self.image = None
        self.log = []
        self.class_name_dic = {0: '正常', 1: '不导电', 2: '擦花', 3: '角位漏底', 4: '桔皮', 5: '漏底', 6: '起坑', 7: '脏点'}

        # 主窗口
        self.ac_exit.triggered.connect(QApplication.exit)
        self.ac_detect_log.triggered.connect(self.action_detect_log)
        self.ac_defect_log.triggered.connect(self.action_defect_log)
        self.ac_update_model.triggered.connect(self.action_update_model)
        self.ac_open_website.triggered.connect(self.action_open_website)
        self.pb_detect.clicked.connect(self.slot_detect)
        self.pb_file_browse.clicked.connect(self.slot_file_browser)
        self.pb_open_file.clicked.connect(self.slot_open_file)
        self.pb_load_model.clicked.connect(self.slot_load_model)
        self.pb_close_model.clicked.connect(self.slot_close_model)
        self.pb_save_image.clicked.connect(self.slot_save_image)
        self.pb_close_model.setDisabled(True)
        self.pb_detect.setDisabled(True)
        self.pb_save_image.setDisabled(True)
        # self.table_history.horizontalHeaderItem(0).setBackground(QBrush(QColor(0,0,0))

        # print 输出重定向
        sys.stdout = EmittingStream(textWritten=self.my_output)
        sys.stderr = EmittingStream(textWritten=self.my_output)

        self.statusBar().showMessage('   就绪')

    def slot_file_browser(self):
        """
        打开文件浏览器槽函数
        :return:
        """
        # 打开文件浏览器，获得选择的文件
        file_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'JPEG(*.jpg) ;; PNG(*.png)')
        # 判断是否选择了文件
        if file_name[0] != '':
            # 显示文件名
            self.le_file.setText(file_name[0])
            self.slot_open_file()

    def slot_open_file(self):
        """
        打开文件槽函数
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

    def slot_detect(self):
        """
        检测槽函数
        :return:
        """
        img_path = self.le_file.text()
        if img_path == '':
            QMessageBox.information(self, 'Error', '请选择文件')
            return
        pre, run_time = predict.predict(img_path)
        # 在历史记录表中插入信息
        row_count = self.table_history.rowCount()
        self.table_history.insertRow(row_count)
        self.table_history.setItem(row_count, 0, QTableWidgetItem(str(row_count+1)))
        self.table_history.item(row_count, 0).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.table_history.setItem(row_count, 1, QTableWidgetItem(str(pre)))
        # 记录此次检测的信息
        temp_log = [img_path, str(pre), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]
        self.log.append(temp_log)
        if pre == -1:
            return
        self.le_class.setText(str(pre))
        self.le_time.setText(str(run_time))
        self.put_text(pre)

    def slot_load_model(self):
        """
        加载模型槽函数
        :return:
        """
        self.refresh_para()
        model = self.cb_model.currentText()
        flag = predict.load_model(model)
        if flag == -1:
            predict.close_sess()
            return
        self.pb_load_model.setDisabled(True)
        self.pb_detect.setEnabled(True)
        self.pb_close_model.setEnabled(True)
        self.le_image_size.setDisabled(True)

    def slot_close_model(self):
        """
        关闭模型槽函数
        :return:
        """
        self.pb_close_model.setDisabled(True)
        predict.close_sess()
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

    def action_detect_log(self):
        """
        打开检测记录文件窗口
        :return:
        """
        self.detect_dialog = LogDialog.DetectLog(self.log)
        self.detect_dialog.show()
        self.detect_dialog.exec()

    def action_defect_log(self):
        """
        打开缺陷记录文件窗口
        :return:
        """
        self.defect_dialog = LogDialog.DefectLog(self.log)
        self.defect_dialog.show()
        self.defect_dialog.exec()

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
            request.urlretrieve(url, '1.jpg')
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
        更新图片尺寸参数
        :return:
        """
        predict.IMG_SIZE = int(self.le_image_size.text())
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
        qp.setFont(QFont('Microsoft YaHei', 80))
        qp.drawText(200, 200, self.class_name_dic[pre])
        qp.end()
        self.lb_image.setPixmap(QPixmap.fromImage(self.image))


class EmittingStream(QtCore.QObject):
    """
    输出的自定义信号类
    """
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

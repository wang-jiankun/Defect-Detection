"""
缺陷检测QT软件--检测记录对话框类
author: 王建坤
date: 2018-10-16
"""
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QHeaderView, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import pandas as pd
import pymysql


class DetectLog(QDialog):
    """
    检测记录类
    """
    def __init__(self, *args):
        super(DetectLog, self).__init__(*args)
        loadUi('ui_detect_log.ui', self)

        # 连接数据库
        self.connection = pymysql.connect(host='localhost', user='root', password='1234',
                                          db='detection_data', charset='utf8')
        self.cursor = self.connection.cursor()
        # 检索数据库中所有检测记录
        sql = "SELECT path,time,detect_class FROM detection_log"
        self.cursor.execute(sql)
        self.detect_logs = self.cursor.fetchall()

        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.table_log.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # self.table_log.horizontalHeader().setStyleSheet("QHeaderView::section{background-color:lightgray;};")
        # self.table_log.verticalHeader().setStyleSheet("QHeaderView::section{background-color:lightgray;};")
        self.pb_clear.clicked.connect(self.slot_clear)
        self.pb_save.clicked.connect(self.slot_save)

        # 生成缺陷记录表
        rows = len(self.detect_logs)
        self.table_log.setRowCount(rows)
        for i in range(rows):
            self.table_log.setItem(i, 0, QTableWidgetItem(str(self.detect_logs[i][0])))
            self.table_log.setItem(i, 1, QTableWidgetItem(str(self.detect_logs[i][1])))
            self.table_log.setItem(i, 2, QTableWidgetItem(self.detect_logs[i][2]))

    def insert_row(self):
        """
        往表格中插入一行
        :return:
        """
        row_count = self.table_log.rowCount()
        self.table_log.insertRow(row_count)

    def slot_clear(self):
        """
        清空表格
        :return:
        """
        self.table_log.clearContents()
        self.detect_logs = []

    def slot_save(self):
        """
        保存检测记录的表格
        :return:
        """
        file_name = QFileDialog.getSaveFileName(self, 'save file', 'log', '.csv')
        detect_csv = pd.DataFrame({'图片路径': [x[0] for x in self.detect_logs],
                                   '检测结果': [x[1] for x in self.detect_logs],
                                   '检测时间': [x[2] for x in self.detect_logs]})
        if file_name[0] != '':
            detect_csv.to_csv(file_name[0])


class DefectLog(QDialog):
    """
    缺陷记录类
    """
    def __init__(self):
        super(DefectLog, self).__init__()
        loadUi('ui_defect_log.ui', self)

        # self.name_class_dic = {'正常': 0, '不导电': 1, '划痕': 2, '污渍': 3, '桔皮': 4, '漏底': 5, '起坑': 6, '脏点': 7}
        self.name_class_dic = {'normal': 0, 'nothing': 1, 'lack_cotton': 2, 'lack_piece': 3, 'wire_fail': 4}
        # 连接数据库
        self.connection = pymysql.connect(host='localhost', user='root', password='1234',
                                          db='detection_data', charset='utf8')
        self.cursor = self.connection.cursor()
        # 检索数据库中所有检测为缺陷的记录
        sql = "SELECT path,time,detect_class FROM detection_log WHERE detect_class != 'normal'"
        self.cursor.execute(sql)
        self.defect_logs = self.cursor.fetchall()

        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.table_defect.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # self.table_defect.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_defect.resizeColumnsToContents()
        self.table_statistics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.pb_save.clicked.connect(self.slot_save)

        # 生成缺陷记录表
        defect_rows = len(self.defect_logs)
        self.table_defect.setRowCount(defect_rows)
        for i in range(defect_rows):
            self.table_defect.setItem(i, 0, QTableWidgetItem(str(self.defect_logs[i][0])))
            self.table_defect.setItem(i, 1, QTableWidgetItem(str(self.defect_logs[i][1])))
            self.table_defect.setItem(i, 2, QTableWidgetItem(self.defect_logs[i][2]))

        # 生成缺陷统计表
        self.class_num_list = [0] * 4
        # 统计不同缺陷的个数，并填充到相应的单元格
        for j in self.defect_logs:
            defect_class = self.name_class_dic[j[2]]
            if -1 < defect_class < 5:
                self.class_num_list[defect_class-1] += 1
        # 计算各缺陷的占比，并填充到相应的单元格
        for k in range(4):
            self.table_statistics.setItem(k, 1, QTableWidgetItem(str(self.class_num_list[k-1])))
            if sum(self.class_num_list):
                self.table_statistics.setItem(k, 2, QTableWidgetItem('%.2f' % (self.class_num_list[k-1]/sum(self.class_num_list))))

    def slot_save(self):
        """
        保存缺陷记录的表格
        :return:
        """
        file_name = QFileDialog.getSaveFileName(self, 'save file', 'log', '.csv')
        if file_name[0] == '':
            return
        if self.tab_table.currentIndex() == 0:
            defect_csv = pd.DataFrame({'图片路径': [x[0] for x in self.defect_logs],
                                       '缺陷类别': [x[1] for x in self.defect_logs],
                                       '检测时间': [x[2] for x in self.defect_logs]})
            defect_csv.to_csv(file_name[0])
        else:
            statistics_csv = pd.DataFrame({self.table_statistics.horizontalHeaderItem(0).text(): [i for i in range(5)], self.table_statistics.horizontalHeaderItem(1).text(): self.class_num_list, self.table_statistics.horizontalHeaderItem(2).text(): [self.table_statistics.itemAt(i, 3).text() for i in range(5)]})
            statistics_csv.to_csv(file_name[0], index=False)

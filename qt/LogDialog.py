"""
缺陷检测QT软件--检测记录对话框类
author: 王建坤
date: 2018-10-16
"""
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QHeaderView, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import pandas as pd


class DetectLog(QDialog):
    """
    检测记录类
    """
    def __init__(self, log_list, *args):
        super(DetectLog, self).__init__(*args)
        loadUi('ui_detect_log.ui', self)

        self.defect_list = log_list
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.table_log.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pb_clear.clicked.connect(self.slot_clear)
        self.pb_save.clicked.connect(self.slot_save)

        rows = len(self.defect_list)
        self.table_log.setRowCount(rows)
        for i in range(rows):
            self.table_log.setItem(i, 0, QTableWidgetItem(self.defect_list[i][0]))
            self.table_log.setItem(i, 1, QTableWidgetItem(self.defect_list[i][1]))
            self.table_log.setItem(i, 2, QTableWidgetItem(self.defect_list[i][2]))

    def insert_row(self):
        """
        插入一行
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
        self.defect_list = []

    def slot_save(self):
        """
        保存检测记录的表格
        :return:
        """
        file_name = QFileDialog.getSaveFileName(self, 'save file', 'log', '.csv')
        detect_csv = pd.DataFrame({'图片路径': [x[0] for x in self.defect_list],
                                   '检测结果': [x[1] for x in self.defect_list],
                                   '检测时间': [x[2] for x in self.defect_list]})
        if file_name[0] != '':
            detect_csv.to_csv(file_name[0])


class DefectLog(QDialog):
    """
    缺陷记录类
    """
    def __init__(self, log_list):
        super(DefectLog, self).__init__()
        loadUi('ui_defect_log.ui', self)

        self.detect_list = log_list
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.table_defect.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pb_save.clicked.connect(self.slot_save)

        # 找出缺陷的记录
        self.defect_list = []
        for sample in self.detect_list:
            if sample[1] == '-1':
                continue
            self.defect_list.append(sample)

        # 缺陷记录表
        defect_rows = len(self.defect_list)
        self.table_defect.setRowCount(defect_rows)
        for i in range(defect_rows):
            self.table_defect.setItem(i, 0, QTableWidgetItem(self.defect_list[i][0]))
            self.table_defect.setItem(i, 1, QTableWidgetItem(self.defect_list[i][1]))
            self.table_defect.setItem(i, 2, QTableWidgetItem(self.defect_list[i][2]))

        # 缺陷统计表
        self.class_num_list = [0] * 12
        for j in self.defect_list:
            if -1 < int(j[1]) < 12:
                self.class_num_list[int(j[1])] += 1
        for k in range(len(self.class_num_list)):
            self.table_statistics.setItem(k, 1, QTableWidgetItem(str(self.class_num_list[k])))
            if sum(self.class_num_list):
                self.table_statistics.setItem(k, 2, QTableWidgetItem('%.2f' %
                                                                     (self.class_num_list[k]/sum(self.class_num_list))))

    def slot_save(self):
        """
        保存缺陷记录的表格
        :return:
        """
        file_name = QFileDialog.getSaveFileName(self, 'save file', 'log', '.csv')
        if file_name[0] == '':
            return
        if self.tab_table.currentIndex() == 0:
            defect_csv = pd.DataFrame({'图片路径': [x[0] for x in self.defect_list],
                                       '缺陷类别': [x[1] for x in self.defect_list],
                                       '检测时间': [x[2] for x in self.defect_list]})
            defect_csv.to_csv(file_name[0])
        else:
            statistics_csv = pd.DataFrame({self.table_statistics.horizontalHeaderItem(0).text(): [i for i in range(12)],
                                           self.table_statistics.horizontalHeaderItem(1).text(): self.class_num_list,
                                           self.table_statistics.horizontalHeaderItem(2).text():
                                               [self.table_statistics.itemAt(i, 3).text() for i in range(12)]})
            statistics_csv.to_csv(file_name[0], index=False)

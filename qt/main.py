"""
缺陷检测QT软件--启动
author: 王建坤
date: 2018-9-25
"""
import sys
from PyQt5.QtWidgets import QApplication
from qt import MainWindow

# 实例化一个 App
app = QApplication(sys.argv)
# 实例化一个 窗口
login = MainWindow.Detect()
# 显示窗口
login.show()
print('Welcome to use defect detection system.')
# 进入主循环
sys.exit(app.exec())




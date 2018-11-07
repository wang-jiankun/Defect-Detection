import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QWidget
from PyQt5.QtCore import QObject, Qt, pyqtSignal


class MySignal(QObject):
    instance = None
    signal = pyqtSignal()
    status_signal = pyqtSignal(str)

    @classmethod
    def my_signal(cls):
        if cls.instance:
            return cls.instance
        else:
            obj = cls()
            cls.instance = obj
            return cls.instance

    def em(self):
        print(id(self.signal))
        self.signal.emit()

    def status_emit(self, s):
        self.status_signal.emit(s)


class MyPushButton(QPushButton):
    def __init__(self, *args):
        super(MyPushButton, self).__init__(*args)

        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        MySignal.my_signal().status_emit('X:'+str(event.pos().x())+' Y:'+str(event.pos().y()))
        self.update()


class MainWindow(QMainWindow):
    """
    主窗口类
    """
    Signal = MySignal.my_signal().signal
    Status_signal = MySignal.my_signal().status_signal

    print(id(Signal), '1')

    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)

        # 设置主窗口的标题及大小
        self.setWindowTitle('主窗口')
        self.resize(400, 300)

        # 创建按钮
        self.btn = MyPushButton(self)
        self.btn.setText('自定义按钮')
        self.btn.move(50, 50)
        self.btn.clicked.connect(self.show_dialog)

        # 自定义信号绑定
        self.Signal.connect(self.test)
        self.Status_signal.connect(self.show_status)

        self.dialog = Dialog()

    def show_dialog(self):
        self.dialog.show()
        self.dialog.exec()

    def test(self):
        self.btn.setText('我改变了')

    def show_status(self, s):
        self.statusBar().showMessage(s)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Home:
            print('Home')
        else:
            QWidget.keyPressEvent(self, event)


class Dialog(QDialog):
    """
    对话框类
    """
    def __init__(self, *args):
        super(Dialog, self).__init__(*args)

        # 设置对话框的标题及大小
        self.setWindowTitle('对话框')
        self.resize(200, 200)
        self.setWindowModality(Qt.ApplicationModal)
        self.btn = QPushButton(self)
        self.btn.setText('改变主窗口按钮的名称')
        self.btn.move(50, 50)
        self.btn.clicked.connect(MySignal.my_signal().em)
        print(id(MySignal.my_signal().signal))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    demo.show()
    sys.exit(app.exec())



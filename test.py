"""
局部代码测试
@Author: JK_Wang
@Time: 20-May-19
"""
import pymysql

connection = pymysql.connect(host='localhost', user='root', password='123456', db='detection', charset='utf8')
cursor = connection.cursor()
sql = "INSERT INTO detection_log(detect_class, path) VALUES(%s, %s)"
cursor.execute(sql, ('1', '2'))
connection.commit()

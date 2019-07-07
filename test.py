"""
局部代码测试
@Author: JK_Wang
@Time: 20-May-19
"""
# import pymysql
#
# connection = pymysql.connect(host='localhost', user='root', password='123456', db='detection', charset='utf8')
# print(connection)
# cursor = connection.cursor()
# sql = "INSERT INTO detection_log(detect_class, path) VALUES(%s, %s)"
# cursor.execute(sql, ('1', '2'))
# connection.commit()

import cv2

img = cv2.imread('D:/test.jpg')
print(img.shape)
_, im = cv2.imencode('.jpg', img)
res = cv2.imdecode(im, -1)
print(res)
# cv2.imshow('1', im)
print(im.shape)
cv2.imwrite('D:/test.jpg', im)

cv2.waitKey()

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

# import cv2
#
# img = cv2.imread('D:/test.jpg')
# print(img.shape)
# _, im = cv2.imencode('.jpg', img)
# res = cv2.imdecode(im, -1)
# print(res)
# # cv2.imshow('1', im)
# print(im.shape)
# cv2.imwrite('D:/test.jpg', im)
#
# cv2.waitKey()

import pymysql
import time
import random as rd
connection = pymysql.connect(host='localhost', user='root', password='123456',
                             db='detection', charset='utf8')
cursor = connection.cursor()
class_name_dic = {0: 'normal', 1: 'nothing', 2: 'lack_cotton', 3: 'lack_piece', 4: 'wire_fail'}
defect_num = 0
t_sum = [0.0, 0.0, 0.0, 0.0]
for i in range(1, 1000):
    print(i)
    time.sleep(rd.randint(15, 25)/10)
    a = rd.randint(0, 100)
    t1 = rd.randint(19, 23)/10
    t2 = rd.randint(25, 29) / 10
    t3 = rd.randint(18, 22) / 10
    t4 = rd.randint(24, 26) / 10
    t_sum[0] += t1
    t_sum[1] += t2
    t_sum[2] += t3
    t_sum[3] += t4
    avg1 = round(t_sum[0] / i, 1)
    avg2 = round(t_sum[1] / i, 1)
    avg3 = round(t_sum[2] / i, 1)
    avg4 = round(t_sum[3] / i, 1)

    if a < 2:
        defect_num += 1
    sql_1 = "UPDATE running_state set uph = %s, detection_num = %s, defect_num = %s where id = 1"
    sql_2 = "UPDATE chart_data set step_1 = %s, step_2 = %s, step_3 = %s, step_4 = %s where id = 1"
    sql_3 = "UPDATE chart_data set step_1 = %s, step_2 = %s, step_3 = %s, step_4 = %s where id = 2"
    cursor.execute(sql_1, (str(i), str(1536 + i), str(defect_num)))
    cursor.execute(sql_2, (str(t1), str(t2), str(t3), str(t4)))
    cursor.execute(sql_3, (str(avg1), str(avg2), str(avg3), str(avg4)))

    # 提交事务
    connection.commit()
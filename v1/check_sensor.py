import tensorflow as tf
import numpy as np
import xlrd

batch_size = 1
eposides = 1000
learning_rate = 0.01
e = 0.1

# 输入数据
x1 = xlrd.open_workbook("data.xlsx")
sheet1 = x1.sheet_by_index(0)
Global_data = sheet1.col_values(0)
GPS_data = sheet1.col_values(1)
delta_data = sheet1.col_values(2)

# 占位符
GPS = tf.placeholder(dtype=tf.float32, shape=[1])
Global = tf.placeholder(dtype=tf.float32, shape=[1])
delta = tf.placeholder(dtype=tf.float32, shape=[1])

# 变量
count = tf.get_variable(shape=[1], dtype=tf.float32, initializer=None, name="count")
k = tf.get_variable(shape=[1], dtype=tf.float32, initializer=None, name="k")
_count = tf.get_variable(shape=[1], dtype=tf.float32, initializer=None, name="ccount")
height = tf.get_variable(shape=[1], dtype=tf.float32, initializer=None, name="height")

# 构建网络
w1 = tf.get_variable(shape=[1, 24], dtype=tf.float32, name="w1")
merge1 = tf.add_n([tf.multiply(GPS, w1), tf.multiply(Global, w1), tf.multiply(delta, w1)])
layer1 = tf.nn.relu(merge1)

w2 = tf.get_variable(shape=[24, 24], dtype=tf.float32, name="w2")
merge2 = tf.matmul(layer1, w2)
layer2 = tf.nn.relu(merge2)

w3 = tf.get_variable(shape=[24, batch_size], dtype=tf.float32, name="w3")
merge3 = tf.matmul(layer2, w3)
height_corrected = tf.nn.relu(merge3)


# 构建损失函数
# 先单纯使用均方差尝试一下
def f1():
    height = GPS
    return height


def f2():
    height = Global
    _count = 0
    return height


def f3():
    tf.add(_count, 1)
    return tf.cond(_count[0] < count[0], f1, f2)


height_corrected = tf.case([((GPS[0] < Global[0] + 3 * delta[0]) & (GPS[0] > Global[0] + 3 * delta[0]), f1),
                    ((GPS[0] > Global[0] + k[0] * delta[0]) | (GPS[0] < Global[0] - k[0] * delta[0]), f2)],
                    default=f3, exclusive=True)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(height - height_corrected)))
update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(eposides):
        for _GPS_data, _Global_data, _delta_data in zip(GPS_data, Global_data, delta_data):
            sess.run([update, height, loss, count, k, _count, height], feed_dict={GPS: [_GPS_data], Global: [_Global_data], delta: [_delta_data]})
            print("k = ", sess.run(k), " count = ", sess.run(count))

        # if i % 50 == 0:
        #     print("k = ", sess.run(k), " count = ", sess.run(count))

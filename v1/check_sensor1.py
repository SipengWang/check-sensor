import tensorflow as tf
import numpy as np
import xlrd

batch_size = 1
eposides = 10000
learning_rate = 0.01
e = 0.1

# 输入数据
x1 = xlrd.open_workbook("data.xlsx")
sheet1 = x1.sheet_by_index(0)
Global_data = sheet1.col_values(0)
GPS_data = sheet1.col_values(1)
delta_data = sheet1.col_values(2)

# 占位符
GPS = tf.placeholder(dtype=tf.float32)
Global = tf.placeholder(dtype=tf.float32)
delta = tf.placeholder(dtype=tf.float32)

print(GPS, Global, delta)

# 变量
count = tf.Variable(dtype=tf.float32, initial_value=10)
k = tf.Variable(dtype=tf.float32, initial_value=10)
_count = tf.Variable(dtype=tf.float32, initial_value=0)
height_corrected = tf.Variable(dtype=tf.float32, initial_value=0)
# height = tf.Variable(dtype=tf.float32, initial_value=0)
print(count)


# # 构建网络
# w1 = tf.get_variable(shape=[1, 24], dtype=tf.float32, name="w1")
# merge1 = tf.add_n([tf.multiply(GPS, w1), tf.multiply(Global, w1), tf.multiply(delta, w1)])
# layer1 = tf.nn.relu(merge1)
#
# w2 = tf.get_variable(shape=[24, 24], dtype=tf.float32, name="w2")
# merge2 = tf.matmul(layer1, w2)
# layer2 = tf.nn.relu(merge2)
#
# w3 = tf.get_variable(shape=[24, batch_size], dtype=tf.float32, name="w3")
# merge3 = tf.matmul(layer2, w3)
# height_corrected = tf.nn.relu(merge3)


# 构建损失函数
# 先单纯使用均方差尝试一下
def f1():
    return GPS


def f2():
    _count = 0
    return Global + k * delta


def f3():
    tf.add(_count, 1)
    return tf.cond(_count < count, f1, f2)


def f4():
    return tf.cond(condition2, f2, f3)


height_corrected = tf.case({(GPS < Global + 3 * delta) & (GPS > Global - 3 * delta): f1,
                            (GPS > Global + k * delta) | (GPS < Global - k * delta): f2},
                           default=f3, exclusive=True)
height_corrected = Global + 1 * k * delta

condition1 = (GPS < Global + 3 * delta) & (GPS > Global - 3 * delta)
condition2 = (GPS > Global + k * delta) | (GPS < Global - k * delta)

# loss = tf.reduce_mean(tf.square(Global - height_corrected))
loss = tf.reduce_mean(tf.cond(condition1, f1, f4))
update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(eposides):
        for _GPS_data, _Global_data, _delta_data in zip(GPS_data, Global_data, delta_data):
            up, lo, hei, _cou = sess.run([update, loss, height_corrected, _count],
                                   feed_dict={GPS: _GPS_data, Global: _Global_data, delta: _delta_data})
            # print("k = ", sess.run(k), " count = ", sess.run(count), "_count = ", sess.run(_count))
            # print("loss = ", lo, "height_corrected = ", hei)

        if i % 50 == 0:
            print(str(i) + "/" + str(eposides), "k = ", sess.run(k),
                  " count = ", sess.run(count), "loss = ", lo, "_count = ", _cou)

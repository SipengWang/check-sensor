import tensorflow as tf


def f1():
    print("f1")
    return tf.constant(1)


def f2():
    print("f2")
    return tf.constant(2)


def f3():
    print("f3")
    return tf.constant(3)

# result = tf.Variable(initial_value=-1, dtype=tf.int32)
result = tf.case({tf.greater(tf.constant(1), tf.constant(2)): f1,
                  tf.less(tf.constant(1), tf.constant(2)): f2}, default=f3, exclusive=True)

tf.Session().run(tf.global_variables_initializer())
print(tf.Session().run(result))
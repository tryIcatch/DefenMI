import tensorflow as tf

def generate_random_tensor(minval, maxval, shape, dtype=tf.float32):
    return tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)

def get_random_tensor(minval, maxval, shape, dtype=tf.float32):
    # 通过 shape=None 创建随机张量
    c1 = generate_random_tensor(minval, maxval, shape=None, dtype=dtype)
    c2 = generate_random_tensor(minval, maxval, shape=None, dtype=dtype)
    c3 = generate_random_tensor(minval, maxval, shape=None, dtype=dtype)

    # 手动添加 batch 维度（shape=(None, ...)）
    c1 = tf.expand_dims(c1, axis=0)
    c2 = tf.expand_dims(c2, axis=0)
    c3 = tf.expand_dims(c3, axis=0)

    return c1, c2, c3


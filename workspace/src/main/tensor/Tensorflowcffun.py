import tensorflow as tf

def fuse(p1, p2):
    q_on = p1*p2
    q_off = (1-p1)*(1-p2)
    return q_on/(q_on+q_off)

def invsig(x):
    return -tf.math.log((1.0/x)-1)

def lag(M, n):
    tiled = tf.tile(M[0], tf.constant([n]))
    tiled = tf.reshape(tiled,(n, len(M[0])))
    cat = tf.concat((tiled, M), axis=0)
    return cat[:-n]

def outer(a, b):
    return tf.reshape(a,[-1])[...,None]*tf.reshape(b,[-1])[None,...]
import tensorflow as tf

def fuse(p1, p2):
    q_on = p1*p2
    q_off = (1-p1)*(1-p2)
    return q_on/(q_on+q_off)

def invsig(x):
    return -tf.math.log((1.0/x)-1)

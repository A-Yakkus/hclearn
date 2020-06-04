import tensorflow as tf


def boltzmannProbs(W, x):      # RETURNS THE PROBABILITY OF A NODE BEING ON
    if type(x) is int or type(x) is float or x.shape is ():
        mult = tf.multiply(W, x)
    else:
        mult = tf.tensordot(W, x, 1)
    E_on  = tf.negative(mult)       #penalty is the negative of the reward (just to make it look like energy)
    E_off = 0.0*E_on
    Q_on = tf.math.exp(-E_on)       #as energy is negated, we have e^(reward)
    Q_off = tf.math.exp(-E_off)
    P_on = tf.math.divide(Q_on, tf.math.add(Q_on, Q_off))
    # P_on = Q_on / (Q_on + Q_off)
    return P_on.numpy()


def hardThreshold(xs):
    return tf.cast(xs>0.5, tf.float32).numpy()
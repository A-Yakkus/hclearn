import tensorflow as tf
import workspace.src.main.tensor.Tensorflowcffun as cffun

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

def argmaxs(xs):
    return tf.argmax(xs, axis=1)
    #T=xs.shape[0]
    #r=np.zeros(xs.shape)
    #for t in range(0,T):
    #    i=argmax(xs[t,:])
    #    r[t,i]=1
    #return r

def addBias(xs):
    T=xs.shape[0]
    out = tf.concat((xs, tf.ones((T, 1))), axis=1)
    #out = tf.hstack((xs,np.ones((T,1))))
    return out

def stripBias(xs):
    return xs[:, 0:-1]

def trainPriorBias(hids):
    p_null_row = tf.math.reduce_mean(addBias(hids), axis=0)
    p_null_row[p_null_row == 0] = 0.00000000123666123666
    p_null_row[p_null_row == 1] = 0.999999999123666123666
    b_null = cffun.invsig(p_null_row)
    return b_null
import numpy as np
from paths import *
import tensorflow as tf
print("Go me")
##### CAPS ARE NOTES BY M.EVANS. Notes are now in README.md or the hclearn notebook

############################### hintonian stuff starts here


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


def trainPriorBias(hids):      # SEEMS TO CONCATENATE AND NORMALISE THE HIDDEN UNIT VALUES
    p_null_row = mean(addBias(hids),0)   #include predicting 1, as a checksum!
    idx=where(p_null_row==0)                  #tweak to avoid Inf, Nans etc
    p_null_row[idx]=0.00000000123666123666
    idx=where(p_null_row==1)
    p_null_row[idx]=0.999999999123666123666
    b_null = invsig(p_null_row)   
    return b_null

def trainW(obs, hids, WB, N_epochs, alpha):    #training observation weights

    T     = hids.shape[0]
    hids = np.hstack((hids, np.ones((T,1)))) #add (redundant, target) bias
    obs  = np.hstack((obs,  np.ones((T,1)))) #add bias AND global bias [no learning in GB, only in local bias]

    print ("training observations")

    N_CA3 = hids.shape[1]
    N_obs = obs.shape[1]

    WO = np.random.rand(N_CA3,N_obs)-0.5
    
    for epoch in range(0,N_epochs):
       ## e=0   #error

        for t in range(0,T):

            o = obs[t,:]

            #WAKE
            xw = hids[t,:]
            C = np.outer(xw,o)
            WO += alpha*C
          

            #SLEEP  -- as 1-step CD unlearning
            b   = boltzmannProbs(WB, 1.0)
            px  = fuse(b, boltzmannProbs(WO,o))          #probs for x next

            xs = (px > np.random.random(px.shape)).astype('d')    #sleep sample (at temp=1)
            po = boltzmannProbs(WO.transpose(), xs) 
            os = (po > np.random.random(po.shape)).astype('d')    #sleep sample (at temp=1)

            C = np.outer(xs,os)

            WO -= alpha*C

            if (t%100)==0:
                obs_hat = hardThreshold(boltzmannProbs(WO.transpose(), hids.transpose()).transpose())
                e = sum((obs_hat[t-100:t,0:-2] - obs[t-100:t,0:-2])**2)/100    #implicit sums over both axes; includes making bias=1 (unnecessary)
                #print (t,e)

        print (epoch, (e+0.0)/T)   #test if it reduces -- pritns average number that were wrong (get down to 4)
    return WO

def argmaxs(xs):
    T=xs.shape[0]
    r=np.zeros(xs.shape)
    for t in range(0,T):
        i=argmax(xs[t,:])
        r[t,i]=1
    return r

def hardThreshold(xs):
    T=xs.shape[0]
    r=np.zeros(xs.shape)
    for t in range(0,T):
        r[t,:]= (xs[t,:]>0.5)
    return r



def addBias(xs):
    T=xs.shape[0]
    out = np.hstack((xs,np.ones((T,1))))
    return out

def stripBias(xs):
    return xs[:, 0:-1]


#hardThreshold(boltzmannProbs(WO.transpose(), hids.transpose()).transpose())


print("[Paths]Go Me!")
from cffun import *
from makeMaze import Senses
from location import Location
from DGStateAlan import DGState, smartCollapse

#Senses, makeMaze

#A path is a log of random walk locations and sensations.
#CHANGED FROM PATH TO PATHS AS IPYTHON DOESN'T LIKE FILES/DIRS CALLED PATH
from Ecstate import ECState
from caonestate import CA1State
from cathreestate import CA3State


class Paths:
    
    def __init__(self, dictNext, N_mazeSize, T_max):
        self.N_mazeSize = N_mazeSize
        self.posLog        = np.zeros((T_max, 3))  #for training, ground truth states. Includes lightState
        self.lightAheadLog = np.zeros((T_max, 1))  #true if there is a light ahead of the agent
        self.lightStateLog = np.zeros((T_max, 1))  #true if there is a light ahead of the agent


        s=[3,3,0]  #state (of agent only).  start at center, facing towards east.
        lightState=0        #there are 4 lights which move when agent reaches NESW respectively

        for t in range(0,T_max):

            if s[0]==2*N_mazeSize and lightState==0: #E arm
                lightState=1 
                #print ("light to N")
            if s[1]==2*N_mazeSize and lightState==1: #N arm
                lightState=2
                #print ("light to W")
            if s[0]==0 and lightState==2: #W
                lightState=3
                #print ("light to S")
            if s[1]==0 and lightState==3: #S
                lightState=0 
                #print ("light to E")
            self.lightStateLog[t] = lightState


            if s[2]==lightState:            #agent facing in same direction as the lit arm
                self.lightAheadLog[t] = 1   #there is a visible light ahead

            self.posLog[t,0:3]=s

            s_nexts = dictNext[tuple(s)]          #possible next locations
            i = random.randrange(0,len(s_nexts))  #choose a random next location
            s = s_nexts[i]


    def getGroundTruthFiring(self,dictSenses,dictGrids,N_mazeSize,t,dghelper=None):

         loc        = self.posLog[t,:]
         lightState = self.lightStateLog[t,0]     #which physical light (eg 
         lightAhead = self.lightAheadLog[t,0]
         senses = dictSenses[tuple(loc)]    
         #HOOK include SURF features in dictSenses structure
         ecState = ECState((senses, lightAhead))
         dgState = DGState(ecState, dictGrids, dghelper)
         ca3State = CA3StateFromInputs(ecState, dgState, lightState) #ideal state, No need to know what a surf feature is...

         if t==0:
             odom=np.zeros((1,2))
         else:
             odom = self.posLog[t,:]-self.posLog[t-1,:]
         return (ecState,dgState,ca3State,odom)


    #makes a "data" matrix, with cols of IDEAL EC and DG outputs. (No noise, and perfect GPS).
    #also returns the ideal CA3 output (used for training) and the raw positions.
    def getGroundTruthFirings(self, dictSenses, dictGrids, N_mazeSize, dghelper=None):
        #print ("get ground truth firings")
        T_max = self.posLog.shape[0]
        ecStates = []  #fill with ECState objects
        dgStates = [] #fill with DGState objects
        ca3States = []
        for t in range(0,T_max):
            (ecState,dgState,ca3State,odom) = self.getGroundTruthFiring(dictSenses,dictGrids,N_mazeSize,t,dghelper) 
            ecStates.append(ecState)           
            dgStates.append(dgState)         
            ca3States.append(ca3State)
        #print ("done")
        return (ecStates, dgStates, ca3States)

    
    #for training only. (Real inference uses noiseless, and adds its own noise AND odometry)
    #this assumes the noise is due to noisy GPS -- not to lost odometry
    def getNoiseyGPSFirings(self, dictSenses, dictGrids, N_mazeSize, dghelper=None):
        T_max = self.posLog.shape[0]
        ecStates = []  #fill with ECState objects
        dgStates = [] #fill with DGState objects
        ca3States = []
        for t in range(0,T_max):
            (ecState,dgState,ca3State,odom) = self.getGroundTruthFiring(dictSenses,dictGrids,N_mazeSize,t,dghelper)
            
            lightState = self.lightStateLog[t,0]

            ecState = ecState.makeNoisyCopy()
            dgState = DGState(ecState, dictGrids, dghelper)
            ca3State = CA3StateFromInputs(ecState, dgState, lightState)

            ecStates.append(ecState)           
            dgStates.append(dgState)         
            ca3States.append(ca3State)
        return (ecStates, dgStates, ca3States)





def CA3StateFromInputs(ec, dg, lightState):
    place    = dg.place.copy()
    hd       = ec.hd.copy()

    place_hd=np.zeros((place.shape[0],hd.shape[0]))
    for i_place in range(0,place.shape[0]):
        for i_hd in range(0, hd.shape[0]):
            place_hd[i_place,i_hd] = place[i_place]*hd[i_hd]

    light = np.zeros(4)  #CA3 light cells. (ie tracking the hidden state of the world)
    #print(lightState) # Jack Stevenson: debug as part of python3 update process
    light[int(lightState)]=1 # Jack Stevenson: convert to integer data type as np.ndarray's can not handle floats.
    
    N_place = place.shape[0] 
    N_light = 4       
    N_hd=4

    light_hd = np.zeros((N_hd, N_light))
    for i_hd in range(0,4):
        for i_light in range(0,N_light):
            light_hd[i_hd,i_light] = light[i_light] * hd[i_hd]
    
    return CA3State(place,place_hd,light,light_hd)
    #return CA3State(place,place_hd,light,light_hd, dg.encodedValues) #ALAN apperantly CA3 doesn't need to know about Surfs? says: path.py line 56 as we are just getting the ground truths? This is backed up by the fact that touch sensors arnt used here

def CA3StateFromVector(v_ca3, N_places):

    N_light=4
    N_hd=4

    place = v_ca3[0:N_places]

    place_hd = v_ca3[N_places:N_places + N_places*4]
    place_hd = place_hd.reshape((N_places,N_hd))

    light = v_ca3[N_places + N_places*4 : N_places + N_places*4 + N_light]   #which of 4 arms is lit

    light_hd = v_ca3[ N_places + N_places*4 + N_light : N_places + N_places*4 + N_light + N_light*N_hd ]
    light_hd = light_hd.reshape((N_hd,N_light)) #TODO check reshape is right way round

    return CA3State(place,place_hd,light,light_hd)


def ca3_states_to_matrix(ca3s):
    T=len(ca3s)
    N=ca3s[0].place.shape[0]
    out = np.zeros((T,N))
    for t in range(0,T):
        out[t,:] = ca3s[t].place
    #TODO convert to x,y coords here?

    return out



#Subbed in mine from DGStateAlan
"""
class DGState:
    def __init__(self, ec, dictGrids):

        N_place = 13
        N_hd = 4       

        l=Location()       #NEW, pure place cells in DG
        l.setGrids(ec.grids, dictGrids)
        self.place=np.zeros(N_place)
        self.place[l.placeId] = 1

        self.hd_lightAhead = np.zeros(4)
        if ec.lightAhead == 1:
            self.hd_lightAhead = ec.hd.copy()

        self.whisker_combis = np.zeros(3)  #extract multi-whisker features. 
        self.whisker_combis[0] = ec.whiskers[0] * ec.whiskers[1] * ec.whiskers[2]   #all on
        self.whisker_combis[1] = (1-ec.whiskers[0]) * (1-ec.whiskers[1]) * (1-ec.whiskers[2])   #none on
        self.whisker_combis[2] = ec.whiskers[0] * (1-ec.whiskers[1]) * ec.whiskers[2]   # both LR walls but no front

        #HOOK, needs to use EC data to define "combis" of features aswell

    def toVector(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead, self.whisker_combis))

    def toVectorSensesOnly(self):
        return np.hstack((self.whisker_combis))

    def toVectorOdomOnly(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead))

    def smartCollapse(self):                         #NEW
        self.place = smartCollapse(self.place)


def smartCollapse(xs):
    idx=argmax(xs)
    r = np.zeros(xs.flatten().shape)
    r[idx]=1
    return r.reshape(xs.shape)
"""





#converts a single place cell vector into an x,y coordinate
def placeCells2placeID(_pcs, n_mazeSize):
    n_places = ((2*n_mazeSize)+1) **2 
    pcs = _pcs.copy()
    pcs=pcs[0:n_places]  #strip down to place cells only
    T = pcs.shape[0]
    grid = pcs.reshape(( ((2*n_mazeSize)+1),   ((2*n_mazeSize)+1) ))
    (xy) = np.where(grid==1)  
    return (xy[0][0], xy[1][0])   #return first (if several) matches






def ca3s2v(ca3s):    #CA3 states to vector
    N = ca3s[0].toVector().shape[0]
    T = len(ca3s)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,0:N]=ca3s[t].toVector()
    return r


##with dentate 
def ecs2vd(ec_states):
    N_ec = ec_states[0].toVector().shape[0]
    N_dg = DGState(ec_states[0]).toVector().shape[0]
    N=N_ec+N_dg
    T = len(ec_states)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,0:N]=ec_states[t].toVectorD()
    return r



##with dentate , senses obly
def ecs2vd_so(ec_states, dictGrids, dghelper=None):
    N = ec_states[0].toVectorSensesOnlyD(dictGrids,dghelper).shape[0]
    T = len(ec_states)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,:]=ec_states[t].toVectorSensesOnlyD(dictGrids,dghelper)
    return r

##with dentate , odom only
def ecs2vd_oo(ec_states, dictGrids):
    N = ec_states[0].toVectorOdomOnlyD(dictGrids).shape[0]
    T = len(ec_states)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,:]=ec_states[t].toVectorOdomOnlyD(dictGrids)
    return r

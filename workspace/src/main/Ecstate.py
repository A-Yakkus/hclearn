from location import Location
from cffun import *
from makeMaze import Senses
from DGStateAlan import DGState


class ECState:           #just flattens grids, and adds lightAhead to the Senses object!
    def __init__(self, arg):
        if isinstance(arg,ECState):  #copy constructor
            self.grids=arg.grids.copy()
            self.hd=arg.hd.copy()
            self.whiskers=arg.whiskers.copy()
            self.rgb=arg.rgb.copy()
            self.lightAhead=arg.lightAhead.copy()
            self.surfs=arg.surfs.copy() #ALAN
        elif isinstance(arg[0], Senses):    #contruct from a (s:Senses, lightAhead:bool) tuple
            senses=arg[0]
            lightAhead=arg[1]
            self.grids=senses.grids.copy()
            self.hd=senses.hd.copy()
            self.whiskers=senses.whiskers.copy()
            self.rgb=senses.rgb.copy()
            self.lightAhead=lightAhead.copy()
            self.surfs=senses.surfs.copy() #ALAN
        elif isinstance(arg[0], np.ndarray):   #TODO test. COnstruct from a (v_ec:vector, nPlaces:int) tuple
            N_grids=arg[1]
            if arg[1]>6:
                pdb.set_trace()
                #print ("ERROR TOO MANY GRIDS!")
            self.grids=arg[0][0:N_grids].reshape((2,N_grids/2))
            self.hd=arg[0][N_grids:N_grids+4]
            self.whiskers=arg[0][N_grids+4:N_grids+4+3]
            self.rgb=arg[0][N_grids+4+3:N_grids+4+3+3]
            self.lightAhead=arg[0][N_grids+4+3+3:N_grids+4+3+3+1]
            #print("HOOK THIS NEED TO BE IMPLEMENTED FOR SURF")

    def collapseToMax(self):    #use this if I was created from a prob vec
        #i=argmax(self.placeCells)
        #self.placeCells*=0
        #self.placeCells[i]=1

        i=argmax(self.hd)
        self.hd*=0
        self.hd[i]=1

        self.whiskers = (self.whiskers>0.5)
        self.rgb = (self.rgb>0.5)
        self.lightAhead = (self.lightAhead>0.5)
        self.surfs = (self.surfs>0.5) #ALAN


    #NB my grids arne't Vechure-style attractors; rahter they passively sum CA1 with odom
    #doign so assumes that the HC output is always right, unless speciafically lost. (Noisy GPS)
    def updateGrids(self, ca1grids, ca1hd, b_odom, N_mazeSize, dictGrids):

        loc=Location()
        loc.setGrids(ca1grids, dictGrids)

        (x_hat_prev, y_hat_prev) = loc.getXY()

        dxys = [[1,0],[0,1],[-1,0],[0,-1]]  #by hd cell
        ihd = argmax(ca1hd)
        odom_dir = dxys[ihd]

        odom = [0,0]
        if b_odom:
            odom=odom_dir

        x_hat_now = x_hat_prev + odom[0]
        y_hat_now = y_hat_prev + odom[1]

        ##SMART UPDATE -- if odom took us outside the maze, then ignore it
        #pdb.set_trace()

        ##if this takes me to somewhere not having a '3'(=N_mazeSize) in the coordinate, then the move was illegal?
        if sum( (x_hat_now==N_mazeSize) + (y_hat_now==N_mazeSize))==0:
            #print ("OFFMAZE FIX: OLD:" ,x_hat_now, y_hat_now)
            x_hat_now = x_hat_prev
            y_hat_now = y_hat_prev
            #print ("NEW:",x_hat_now, y_hat_now)
        x_hat_now = crop(x_hat_now, 0, 2*N_mazeSize)
        y_hat_now = crop(y_hat_now, 0, 2*N_mazeSize)  #restrict to locations in the maze

        loc=Location()
        loc.setXY(x_hat_now, y_hat_now)
        #self.placeCells=zeros(ca1placeCells.shape)
        #self.placeCells[loc.placeId] = 1
        self.grids = loc.getGrids().copy()

    #dth in rads; HDs are four bool cells
    def updateHeading(self, ca1hd, d_th):
        self.hd=np.zeros((4))
        i_old = argmax(ca1hd)
        i_new = (i_old+d_th)%4
        #print(i_new) # debug line
        self.hd[int(i_new)]=1 # Jack Stevenson: Numpy doesn't like floating points as indices :(

    def toVector(self):
        return np.hstack((self.grids.flatten(), self.hd, self.whiskers, self.rgb, self.lightAhead, self.surfs) )

    def toVectorSensesOnly(self):
        senses=  np.hstack((self.whiskers, self.rgb, self.lightAhead, self.surfs) )
        return senses

    def toVectorOdomOnly(self):
        return np.hstack((self.grids.flatten(), self.hd) )

    def toVectorD(self,dictGrids, dghelper=None):  #with dentate and bias
        return np.hstack(( self.toVector(), DGState(self, dictGrids, dghelper).toVector() ))

    def toVectorSensesOnlyD(self,dictGrids, dghelper=None):
        senses = np.hstack((self.toVectorSensesOnly(), DGState(self, dictGrids, dghelper).toVectorSensesOnly()))
        return senses

    def toVectorOdomOnlyD(self,dictGrids):
        return np.hstack((self.toVectorOdomOnly(), DGState(self,dictGrids).toVectorOdomOnly()))

    def toString(self):
        r="EC:\n  grids:"+str(self.grids)+"\n  hd:"+str(self.hd)+"\n  whiskers:"+str(self.whiskers)+"\n  rgb:"+str(self.rgb)+"\n  lightAhead:"+str(self.lightAhead)+"\n surfs:"+str(self.surfs)
        return r

    #GPSnoise:use ONLY to simulate occasional lostness for TRAINING, not during inference
    #(might want to make noisy odom elsewhere for inference)
    def makeNoisyCopy(self, b_GPSNoise=True):   #makes and returns a noisy copy
        ec = ECState(self)

        p_flip = 0.2
        p_flip_odom = 0.2   #testing, make the grids,hds very unreliable (TODO iterative training??)

        if b_GPSNoise:
            if random.random()<p_flip_odom:    #simulate grid errors- fmove to a random place (as when lost)
                N_places = 13
                i = random.randrange(0,N_places)
                loc = Location()
                loc.setPlaceId(i)
                ec.grids = loc.getGrids().copy()

            if random.random()<p_flip_odom:    #simulate HD errors
                i = random.randrange(0,4)
                ec.hd[:] = 0
                ec.hd[i] = 1
            ##if random.random()< 0.05:  ####simulate lost/reset events WRITEUP: EM like estimation of own error rate needed here (cf. Mitch's chanel equalisation decision feedback/decision directed)
            ##    ec.placeCells = 0.0 * ec.placeCells
            ##    ec.hd = 0.0 * ec.hd  ##no this isnt what we want to do -- we dont want to leatn flatness as an OUTPUT!


        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[0] = 1-ec.whiskers[0]
        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[0] = 1-ec.whiskers[0]
        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[1] = 1-ec.whiskers[1]
        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[2] = 1-ec.whiskers[2]
        if random.random()<p_flip:    #flip lightAhead
            ec.lightAhead = 1-ec.lightAhead
        if random.random()<p_flip:    #flip colors
            ec.rgb[0] = 1-ec.rgb[0]
        if random.random()<p_flip:    #flip colors
            ec.rgb[1] = 1-ec.rgb[1]
        for featureInd, feature in enumerate(ec.surfs): #ALAN implemented flipping
            if random.random()<p_flip:
                ec.surfs[featureInd] = 1-feature
        return ec


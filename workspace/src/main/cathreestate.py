import numpy as np
from DGStateAlan import smartCollapse

#God's eye, ideal CA3 response to ideal EC and DG states
class CA3State:
    def __init__(self, place, place_hd, light, light_hd):
        self.place=place
        self.place_hd=place_hd
        self.light=light
        self.light_hd=light_hd  #this is the light STATE not lightAhead
        #self.biases = np.ones((5000, 1)) # Uncomment me to artificially increase size of CA3/rbm.
        #self.surfs=surfs #ALAN not needed because we are looking at ideal?

    def toVector(self):
        return np.hstack(( self.place, self.place_hd.flatten(), self.light, self.light_hd.flatten()))#, self.biases.flatten())) #without Bias

    def toString(self):
        r = "CA3state:\n  place="+str(self.place)+"\n  phace_hd:"+str(self.place_hd)+"\n  light:"+str(self.light)+"\n  light_hd:"+str(self.light_hd)
        #r+= "biases"+str(self.biases)
        return r

    def smartCollapse(self):
        self.place = smartCollapse(self.place)
        self.place_hd = smartCollapse(self.place_hd)
        self.light = smartCollapse(self.light)
        self.light_hd = smartCollapse(self.light_hd)
        #self.biases = smartCollapse(self.biases)
        #self.surfs = smartCollapse(self.encodedValues) #ALAN is this necessary?

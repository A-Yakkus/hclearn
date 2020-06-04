from cffun import *
from DGStateAlan import smartCollapse
from location import Location


class CA1State:
    def __init__(self, p_odom, p_senses, dghelper=None):
        i=0
        n_grids=6
        n_hd=4
        n_places=13

        #pdb.set_trace()

        p_grids  = p_odom[i:i+n_grids];   i+=n_grids
        p_hd     = p_odom[i:i+n_hd];      i+=n_hd
        p_places = p_odom[i:i+n_places];  i+=n_places

        i=0
        n_whiskers=3
        n_rgb=3
        n_lightAhead=1
        n_whiskerCombis=3

        p_whiskers = p_senses[i:i+n_whiskers]; i+=n_whiskers
        p_rgb = p_senses[i:i+n_rgb]; i+=n_rgb
        p_lightAhead = p_senses[i:i+n_lightAhead]; i+=n_lightAhead
        p_whiskerCombis = p_senses[i:i+n_whiskerCombis]; i+=n_whiskerCombis

        #HOOK: put your decoding of output (whatever representation that is...., here)
        #decode remaining sensors which are the features previously encoded
        if dghelper is not None:
            #Get the number of surf features
            n_surfFeatures = dghelper.numOfSurfFeatures
            #Get the number of encoded features
            n_encoded = dghelper.numOfEncodedFeatures
            #print("Num of surf features: %d\nNum of encodedFeatures: %d\nNum of all feautres: %d" % (n_surfFeatures, n_encoded, (n_surfFeatures+n_encoded)))
            p_surfFeatures = p_senses[i:i+n_surfFeatures]; i+=n_surfFeatures
            p_encoded = p_senses[i:i+n_encoded]; i+=n_encoded

            #We now have two sources of surf, one from the probabilities that came from EC into CA3, and one from the DG encoded going into CA3
            #Dumb decode the former:
            surfFromEC = (p_surfFeatures>0.5)

            #Very smart decode... use the weights learnt to decode back to EC space
            surfFromDG = dghelper.decode(p_encoded)

            #Experiment with using both see what advantage DG gives over EC
            self.surfs = surfFromDG

        #print("Total length of senses:%d, used:%d" % (len(p_senses), i))

        #smart decoding, use smart feature collapse, then create ECd pops here too
        self.places = smartCollapse(p_places)
        self.hd = smartCollapse(p_hd)
        #print("p_whiskerCombis: %s" % p_whiskerCombis)
        self.whiskerCombis = smartCollapse(p_whiskerCombis)

        loc=Location()
        loc.setPlaceId(argmax(self.places))
        self.grids=loc.getGrids()

        #dumb decodes
        self.lightAhead = (p_lightAhead>0.5)
        self.rgb = (p_rgb>0.5)

        #print("whisker combis: %s" % self.whiskerCombis)
        #whiskers
        if self.whiskerCombis[0]:
            self.whiskers=np.array([1,1,1])  #all
        elif self.whiskerCombis[1]:
            #print(self.places)
            #print("no whiskers touching")
            self.whiskers=np.array([0,0,0])  #none
        elif self.whiskerCombis[2]:
            #print("left right whiskers touching")
            self.whiskers=np.array([1,0,1])  #L+R


    def toString(self):
        r="CA1:\n  grids:"+str(self.grids)+"\n  hd:"+str(self.hd)+"\n  whiskers:"+str(self.whiskers)+"\n  rgb:"+str(self.rgb)+"\n  lightAhead:"+str(self.lightAhead)+"\n  place:"+str(self.places)+"\n  wcombis:"+str(self.whiskerCombis)
        return r


import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#from hcq import *
#from gui import *
from location import *
from paths import * # CHANGED AS IPYTHON DOESN'T LIKE FILES/DIRS CALLED PATH
from makeMaze import makeMaze
import plotPlaceCells
#from os import sys
import learnWeights
import sys

pathing = "../data/"



if len(sys.argv) == 3:
    b_useNewDG = (sys.argv[1] == "True")
    learningRate = float(sys.argv[2])
else:
    b_useNewDG = True
    learningRate = 0.01


def get_accuracy_naive(hist, path, dict_grids):
    T = len(hist.ca1s)

    xys_bel = np.zeros((T, 2))
    ihds_bel = np.zeros((T, 1))
    for t in range(0, T):
        loc = Location()
        loc.setGrids(hist.ca1s[t].grids, dict_grids)

        xys_bel[t, :] = loc.getXY()
        ihds_bel[t] = np.argmax(hist.ca1s[t].hd)

    a1 = xys_bel
    a2 = path.posLog[:, 0:2]
    a3 = a1 == a2
    cnf = np.zeros((2, 2))
    print(a3.shape)
    for k1 in range(a3.shape[0]):
        if np.all(a3[k1, :] == [True, True]):
            cnf[0, 0]+=1
        if np.all(a3[k1, :] == [True, False]):
            cnf[0, 1]+=1
        if np.all(a3[k1, :] == [False, True]):
            cnf[1, 0]+=1
        if np.all(a3[k1, :] == [False, False]):
            cnf[1, 1]+=1

    print(cnf)


print("Sys:%d" % len(sys.argv))
print("DG:%s" % b_useNewDG)
print("LR:%f" % learningRate)

np.set_printoptions(threshold=sys.maxsize)         #=format short in matlab

N_mazeSize = 3
T = 3000   # trained on 30000   # better to have one long path than mult epochs on overfit little path
b_learnWeights = True
b_plot = False
b_inference = True

# make maze, including ideal percepts at each place
[dictSenses, dictAvailableActions, dictNext] = makeMaze(N_mazeSize, b_useNewDG)
dictGrids = DictGrids()
# a random walk through the maze -- a list of world states (not percepts)
path = Paths(dictNext, N_mazeSize, T)

# ideal percepts for path, for plotting only
(ecs_gnd, dgs_gnd, ca3s_gnd) = path.getGroundTruthFirings(dictSenses, dictGrids, N_mazeSize)

if b_learnWeights:
    print ("TRAINING...")
    # ALAN: Careful this won't exist if b_learnDGWeights is not true (I.e. we're not using SURF features)
    dghelper = learnWeights.learn(path, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=True, b_learnDGWeights=b_useNewDG, learningRate=learningRate)
    #dghelper = learnWeights.learn(path, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=False, b_learnDGWeights=False, learningRate=learningRate)
    #dghelper = learnWeights.learn(path, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=False, b_learnTrained=True, b_learnDGWeights=False, learningRate=learningRate)
    #dghelper = learnWeights.learn(path, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=False, b_learnTrained=False, b_learnDGWeights=True, learningRate=learningRate)
else:
    dghelper=None

"""
## NB loading trained versions from genuine wake sleep
WR_t = np.load(pathing+'tWR.npy')
WO_t = np.load(pathing+'tWO.npy')
WS_t = np.load(pathing+'tWS.npy')
WB_t = np.load(pathing+'tWB.npy')
WB_t = WB_t.reshape(WB_t.shape[0])   # in case was learned as 1*N array instead of just N.

## NB loading trained versions from perfect look-ahead training
WR_ideal = np.load(pathing+'WR.npy')
WO_ideal = np.load(pathing+'WO.npy')
WS_ideal = np.load(pathing+'WS.npy')
WB_ideal = np.load(pathing+'WB.npy')

WR_rand = 0+ 0*np.random.random(WR_ideal.shape)
WB_rand = 0+ 0*np.random.random(WB_ideal.shape)
WO_rand = 0+ 0*np.random.random(WO_ideal.shape)
WS_rand = 0+ 0*np.random.random(WS_ideal.shape)

b_inference = True

b_obsOnly = False
b_usePrevGroundTruthCA3 = False
b_useGroundTruthGrids = False
b_useSub = True
b_learn = False

if b_inference:
    # print ("INFERENCE...")

    random.seed(SEED) ;  np.random.seed(SEED)
    hist1 = makeMAPPredictions(path,dictGrids, dictSenses, WB_t, WR_t, WS_t, WO_t, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Learned", b_learn=b_learn)
    # HOOK test with ground truths on and off

    random.seed(SEED) ;  np.random.seed(SEED)
    hist2 = makeMAPPredictions(path,dictGrids, dictSenses, WB_rand,  WR_rand,  WS_rand, WO_rand, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random", b_learn=b_learn)

    random.seed(SEED) ;  np.random.seed(SEED)
    hist3 = makeMAPPredictions(path,dictGrids, dictSenses, WB_ideal, WR_ideal, WS_ideal, WO_ideal, dghelper, b_obsOnly=b_obsOnly,  b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids, b_useSub=b_useSub, str_title="Handset", b_learn=b_learn)

    print("Hist1....Learned")
    get_accuracy_naive(hist1, path, dictGrids)
    print()
    print("Hist2....Random")
    get_accuracy_naive(hist2, path, dictGrids)
    print()
    print("Hist3....Handset")
    get_accuracy_naive(hist3, path, dictGrids)
    print()
# print ("DONE")


\begin{figure*}[h]
        \includegraphics[width=0.7\textwidth]{figures_and_data/profiling_snapshot.png}
        \caption{Example output from the python profiler module.
        Contains information on number of calls, time spent executing code in that function only and cumulative time
        from the start of the function call until it exits.
        Also respects average time per call to the function.}
    \end{figure*}




if b_plot:

    # #weights are modified in place
    # np.save('tWR',WR_rand)
    # np.save('tWS',WS)
    # np.save('tWB',WB_rand)
    # np.save('tWO',WO)
    anote = "obsonly%s_gndCA3%s_gndgrid%s_sub%s_bl%s_MERGED0102_SUBT026" % (b_obsOnly, b_usePrevGroundTruthCA3, b_useGroundTruthGrids, b_useSub, b_learn)
    
    (lost1,xys1) = plotResults(path, hist1, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost2,xys2) = plotResults(path, hist2, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost3,xys3) = plotResults(path, hist3, dictGrids, b_useNewDG, learningRate, note=anote)

    # savefig('out/run.eps')
 
    plotErrors(hist1, hist2, hist3, lost1, lost2, lost3, learningRate, surfTest=b_useNewDG, note=anote)
 
    # figure()
    # drawMaze()
    # hold(True)
    # drawPath(path.posLog[:,0:2],'k')
    # savefig('out/maze.eps')
    # drawPath(xys_pi, 'b')

    # clf()
    show()

    
    b_other = False
    if b_other:
        # print ("close the first plus maze window to begin the slideshow of place fields!")
        for i in range(0,100):
            (r, visits,firings) = plotPlaceCells(hist1, i, dictGrids)
            clf()
            gray()
            imagesc(r)
            show()

            fn = 'outPC/cell'+str(i)
            savefig(fn)"""



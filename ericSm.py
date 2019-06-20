# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:49:41 2017

@author: hanshalbe
"""
import numpy as np
import mastermind
import pandas as pd

#import os

df = pd.read_csv('mastermindproto2.csv', header=0)

#taskid =int(os.environ['SGE_TASK_ID'])
taskid=451


N=11
s1 = np.logspace(-4, 7, num=N, endpoint=False, base=2.0, dtype=np.float64)
s2 = np.logspace(-4, 7, num=N, endpoint=False, base=2.0, dtype=np.float64)
nn= np.linspace(1, 35, 35)
gn= np.linspace(1, 5, 5)

aG, bG, cG, dG = np.meshgrid(nn, gn, s1, s2)
aG = aG.flatten()
bG = bG.flatten()
cG = cG.flatten()
dG = dG.flatten()

dall=pd.DataFrame({'id':aG, 'game':bG, 's1':cG, 's2':dG})

df=df.loc[df.id==dall.id[taskid], :]
df=df.loc[df.game==dall.game[taskid], :]
df.index = np.arange(0, len(df-1))

game = mastermind.Game(codelength=3,
                       codejar=[df.code1[0],
                                df.code2[0],
                                df.code3[0],
                                df.code4[0],
                                df.code5[0],
                                df.code6[0]], 
                                logging=False)
#
game.initialize(code=[df.truth1[0],
                      df.truth2[0],
                      df.truth3[0]])

    
#
for i in np.arange(len(df)):
    if i == 0:
        fs = game.get_feasible_set()
        ent= game.get_ents(fs, t=dall.s1[taskid], r=dall.s2[taskid])
        prob = game.get_probs()
        first = game.codepool[:,0].flatten()
        second = game.codepool[:,1].flatten()
        third = game.codepool[:,2].flatten()
        outfile=pd.DataFrame({'first': first, 'second': second, 'third': third, 
                              'prob': prob, 'eig': ent})
    
    if i > 0:
        game.guess([df.guess1[i-1],df.guess2[i-1],df.guess3[i-1]])
        fs = game.get_feasible_set()
        ent= game.get_ents(fs, t=dall.s1[taskid], r=dall.s2[taskid])
        prob = game.get_probs()
        first = game.codepool[:,0].flatten()
        second = game.codepool[:,1].flatten()
        third = game.codepool[:,2].flatten()
        dummy=pd.DataFrame({'first': first, 'second': second, 'third': third, 
                            'prob': prob, 'eig': ent})
        frames = [outfile, dummy]
        outfile = pd.concat(frames)

batchname='job'+str(taskid)+'.csv' 
outfile.to_csv(batchname)
'''
Inputs a ROOT file, outputs a csv file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles(2lSS 3l)/{flat, fourVec}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 topReco.py <input root file>
'''

import ROOT
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import rootpy.io
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
from dictTop import topDict2lSS, topDict3l
import functionsMatch
from functionsMatch import selection2lSS, jetCombosTop2lSS, jetCombosTop3l

#Open input file
inf = sys.argv[1]

if '3l' in inf:
    topDict = topDict3l
    is3l = True
elif '2lSS' in inf:
    topDict = topDict2lSS
    is3l = False
else:
    print('Not sure which channel to use')
    exit()

f = ROOT.TFile.Open(sys.argv[1])
nom = f.Get('nominal')

#initialize output dicts
eventsTop = []

#Loop over all entries
nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))

    nom.GetEntry(idx)
    
    topJets = []
    badJets = []

    for i in range(len(nom.jet_pt)):
        if nom.jet_jvt[i]<0.59: continue

        if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthPartonLabel[i])==5:
            topJets.append(i)
        else:
            badJets.append(i)
        
    if len(topJets)!=2: continue

    k = topDict( nom, topJets[0], topJets[1], 1 )
    p = fourVecDict( nom, topJets[0], topJets[1], 1 )

    eventsTop.append(k)

    for l in range(min([3, len(badJets)]) ):
        j = random.sample(badJets,1)[0]
        k = topDict( nom, topJets[0], j, 0 )
        p = fourVecDict( nom, topJets[0], j, 0 ) 
        eventsTop.append(k)

    for l in range(min([3, len(badJets)]) ):
        i = random.sample(badJets,1)[0]
        k = topDict( nom, i, topJets[1], 0 )
        p = fourVecDict( nom, i, topJets[1], 0 )
        eventsTop.append(k)

    for l in range(min([6, len(badJets)]) ):
        if len(badJets)>2:
            i,j = random.sample(badJets,2)
            k = topDict( nom, i, j, 0 ) 
            eventsTop.append(k)

dfTop = pd.DataFrame.from_dict(eventsTop)
dfTop = shuffle(dfTop)

outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
if is3l:
    dfTop.to_csv('csvFiles/top3l/'+outF, index=False)
else:
    dfTop.to_csv('csvFiles/top2lSS/'+outF, index=False)        

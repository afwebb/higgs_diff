''' 
Test how often taking the highest b-tagged jets correctly identifies b-jets from tops
Outputs the percentage of events the model correctly identifies both b-jets, and at least one b-jet
Usage: python3.6 testKerasTop2lSS.py <inputFile> <kerasModel> <kerasNormFactors>
'''

import ROOT
import pandas as pd
import numpy as np
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
import pickle

#Load in the input file
inf = sys.argv[1]
f = ROOT.TFile.Open(inf, "READ")
nom = f.Get('nominal')

events = []
nEntries = nom.GetEntries()

nEvents=0
nCorrect=0
oneCorrect=0

n1b,n2b,n3b=0,0,0
n1bCorrect, n2bCorrect, n3bCorrect = 0,0,0

#Loop over each entry, add to events dict
for idx in range(nEntries):
    #if idx%1000==0:
    #    print(str(idx)+'/'+str(nEntries))
    if idx==5000:
        break

    nom.GetEntry(idx)
    #if nom.nJets_OR_DL1r_70<3: continue

    #Apply 2lSS preselection                                                                                               
    #if not selection2lSS(nom): 
    #    continue

    truthBs = []
    top1, top2 = -1, -1
    btag1, btag2 = -99, -99
    for i in range(len(nom.jet_pt)):
        if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthflav[i])==5:
            truthBs.append(i)
        #if nom.jet_DL1r[i]>btag1:
        if nom.jet_tagWeightBin_DL1r_Continuous[i]>btag1:
            if btag1>btag2:
                top2, btag2 = top1, btag1
            top1 = i
            #btag1 = nom.jet_DL1r[i]
            btag1 = nom.jet_tagWeightBin_DL1r_Continuous[i]
        #elif nom.jet_DL1r[i]>btag2:
        elif nom.jet_tagWeightBin_DL1r_Continuous[i]>btag2:
            top2 = i
            #btag2 = nom.jet_DL1r[i]
            btag2 = nom.jet_tagWeightBin_DL1r_Continuous[i]
 
            #print(btag1, btag2)
    if len(truthBs)!=2: continue

    if nom.nJets_OR_DL1r_70==1:
        n1b+=1
    elif nom.nJets_OR_DL1r_70==2:
        n2b+=1
    else:
        n3b+=1

    topMatches = [top1, top2]
    #print('truth scores: ', [nom.jet_DL1r[x] for x in truthBs], '\t tag scores: ', [nom.jet_DL1r[x] for x in topMatches])
    nEvents+=1
    #if abs(nom.jet_parents[topMatches[0]])==6 and abs(nom.jet_parents[topMatches[1]])==6:
    if topMatches[0] in truthBs and topMatches[1] in truthBs:
        if nom.nJets_OR_DL1r_70==1:                                                                                         
            n1bCorrect+=1
        elif nom.nJets_OR_DL1r_70==2:
            n2bCorrect+=1
        else:                                                                                                           
            n3bCorrect+=1
        nCorrect+=1
    #if abs(nom.jet_parents[topMatches[0]])==6 or abs(nom.jet_parents[topMatches[1]])==6:
    if topMatches[0] in truthBs or topMatches[1] in truthBs:
        oneCorrect+=1
        

print("Both Correct", nCorrect/nEvents)
print("One Correct", oneCorrect/nEvents)
print(f"1b Correct: {str(round(n1bCorrect/n1b, 3))}\n")
print(f"2b Correct: {str(round(n2bCorrect/n2b, 3))}\n")
print(f">=3b Correct: {str(round(n3bCorrect/n3b, 3))}\n")

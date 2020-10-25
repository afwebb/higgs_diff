'''
Inputs a text file listing input root files. 
Outputs a csv for each file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles/top{2lSS 3l}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 topReco.py <list of root files>
'''

import ROOT
from ROOT import TFile
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import rootpy.io
import sys
import math
from math import sqrt
from numpy import unwrap, arange
from rootpy.vector import LorentzVector
import random
from dictTop import topDict2lSS, topDict3l
from joblib import Parallel, delayed
import multiprocessing 
from functionsMatch import selection2lSS, jetCombosTop

def runReco(inf):
    ''' 
    load a root file, loop over each event. Create dataframe containing kinematics of different pairings of jets
    '''

    #Figure out which channel to use
    if '3l' in inf:
        channel='3l'
    elif '2lSS' in inf:
        channel='2lSS'
    else:
        print('Not sure which channel to use')
        exit()
        
    #Open the root file
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    
    #initialize output dicts
    eventsTop = 0
    
    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
        
        nom.GetEntry(idx)
    
        #Separate jets into tops and others, add their indices
        topJets = []
        badJets = []

        #Include all possible combos
        topCombos = jetCombosTop(channel, nom, 1)
        if len(topCombos['flatDicts'])==0:
            continue
        #eventsTop.append(topCombos['flatDicts'])
        #print(topCombos['flatDicts'])
        if eventsTop==0:
            eventsTop=topCombos['flatDicts']
        else:
            for k in eventsTop:                                                           
                eventsTop[k].extend(topCombos['flatDicts'][k])

    # Convert to dataframe, shuffle entries
    dfTop = pd.DataFrame.from_dict(eventsTop)
    dfTop = shuffle(dfTop)

    #Write output to csv file
    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='3l':
        dfTop.to_csv('csvFiles/top3l_15/'+outF, index=False, float_format='%.3f')
    else:
        dfTop.to_csv('csvFiles/top2lSS_15/'+outF, index=False, float_format='%.3f')        

#Run in parallel
linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=12)(delayed(runReco)(inf) for inf in linelist)

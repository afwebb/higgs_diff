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

        '''
        for k in eventsTop:
            eventsTop[k]+=(x[k] for x in topCombos['floatDicts'])

        for i in range(len(nom.jet_pt)):
            if nom.jet_jvt[i]<0.59: continue
            if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthPartonLabel[i])==5:
                topJets.append(i)
            else:
                badJets.append(i)
            
        if len(topJets)!=2: continue #Only consider events where both Bs are reconstructed
        eventsTop.append( topDict( nom, topJets[0], topJets[1], 1 ) ) # Add truth pairing

        # One good jet, one bad
        for l in range(min([3, len(badJets)]) ):
            j = random.sample(badJets,1)[0]
            eventsTop.append( topDict( nom, topJets[0], j, 0 ) )

        # One good, one bad
        for l in range(min([3, len(badJets)]) ):
            i = random.sample(badJets,1)[0]
            eventsTop.append( topDict( nom, i, topJets[1], 0 ) )

        # Both jets bad
        for l in range(min([6, len(badJets)]) ):
            if len(badJets)>2:
                i,j = random.sample(badJets,2)
                eventsTop.append( topDict( nom, i, j, 0 ) )
        '''
    # Convert to dataframe, shuffle entries
    dfTop = pd.DataFrame.from_dict(eventsTop)
    dfTop = shuffle(dfTop)

    #Write output to csv file
    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='3l':
        dfTop.to_csv('csvFiles/top3l/'+outF, index=False, float_format='%.3f')
    else:
        dfTop.to_csv('csvFiles/top2lSS/'+outF, index=False, float_format='%.3f')        

#Run in parallel
linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=10)(delayed(runReco)(inf) for inf in linelist)

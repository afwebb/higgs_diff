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
import root_pandas
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
import functionsMatch
from functionsMatch import selection2lSS, jetCombosTop

def runReco(inf):
    ''' 
    load a root file, loop over each event. Create dataframe containing kinematics of different pairings of jets
    '''

    #Figure out which channel to use
    if '3l' in inf:
        topDict = topDict3l
        channel = '3l'
    elif '2lSS' in inf:
        topDict = topDict2lSS
        channel = '2lSS'
    else:
        print('Not sure which channel to use')
        exit()
        
    #Open the root file
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    
    
    try:
        nom.GetEntries()
    except:
        print("failed to open "+inf)
        return 
    if nom.GetEntries()==0:
        print('no entries found')
        return

    try:
        nom.Mll01
    except:
        print('failed for '+inf)
        return 

    #Loop over all entries
    nEntries = nom.GetEntries()
    eventsTop = pd.DataFrame()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))

        nom.GetEntry(idx)
    
        #Get Dataframe of all input features
        combosTop = jetCombosTop(channel, nom, 0)
        #if len(combosTop['flatDicts'])>0:
        topDF = pd.DataFrame(combosTop['flatDicts'])
        #topDF = combosTop['flatDicts']
        if 'data1' not in inf:
            topDF['weight'] = nom.weight*nom.weight_leptonSF*nom.weight_bTagSF_DL1r_Continuous
            if '41047' in inf or '410389' in inf:
                #print(nom.m_hasMEphoton_DRgt02_nonhad)
                topDF['m_hasMEphoton_DRgt02_nonhad'] =  nom.m_hasMEphoton_DRgt02_nonhad
            topDF['mcChannelNumber'] = nom.mcChannelNumber
        if idx==0:
            eventsTop = topDF
        else:
            eventsTop = eventsTop.append(topDF)
        
    #Write output to root file
    import root_pandas
    #eventsTop = pd.DataFrame(eventsTop)
    outF = '/'.join(inf.split("/")[-2:])
    if channel=='3l':
        eventsTop.to_root('3l/'+outF, key='nominal')
    else:
        eventsTop.to_root('2lSS/'+outF, key='nominal')        

#Run in parallel
linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=30)(delayed(runReco)(inf) for inf in linelist)

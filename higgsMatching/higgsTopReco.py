'''
Inputs a ROOT file, outputs a csv file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles(2lSS 3l)/{flat, fourVec}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 higgsTopReco.py <input root file> <channel>
'''

import pandas as pd
from tensorflow.keras.models import load_model
import ROOT
from ROOT import TFile
from sklearn.utils import shuffle
import numpy as np
import sys
import random
from dictHiggs import higgsTopDict2lSS, higgsTopDict3lS, higgsTopDict3lF
from functionsMatch import jetCombosTop, findBestTopKeras, higgsTopCombos
from joblib import Parallel, delayed
import multiprocessing

#Open input file
#inf = sys.argv[1]
#outDir = sys.argv[2]

def runReco(inf):
    #Set the channel, load in the top model
    if '3l' in inf:
        channel='3l'
        topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy")
        flatDict = higgsTopDict3lS
        is3l = True
    elif '2lSS' in inf:
        channel='2lSS'
        topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top2lSS.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top2lSS_normFactors.npy")
        flatDict = higgsTopDict2lSS
        is3l = False
    else:
        print(f'Channel {channel} is invalid. Should be 2lSS or 3l')
        exit()
        
    print('loaded')
    topMaxVals = topNormFactors[0]                                                                                      
    topMinVals = topNormFactors[1]
    topDiff = topMaxVals - topMinVals
    
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    
    #initialize output dicts
    events = {}
    events3lF = []
    
    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
            
        nom.GetEntry(idx)

        #Check if the Higgs decay products are reconstructed - first leptons
        if channel=='2lSS' and nom.lep_Parent_0!=25 and nom.lep_Parent_1!=25:
            continue
        if is3l: # Check for lepton, decide if event is 3lF or 3lS
            if nom.lep_Parent_1!=25 and nom.lep_Parent_2!=25:
                continue
            if nom.lep_Parent_0 == 25: 
                channel='3lF'
            else: 
                channel='3lS'

        #Check if Higgs jets are reconstructed
        if channel!='3lF' and sum([x==25 for x in nom.jet_parents])!=2: continue
        #if sum([x==25 for x in nom.jet_parents])!=2: continue

        #Find the b-jets from tops
        topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
        if not topRes:
            continue
        topIdx0, topIdx1 = topRes['bestComb']                                                                         
        topScore = topRes['topScore']

        #Get all possible combinations
        combos = higgsTopCombos(channel, nom, topIdx0, topIdx1, topScore, 1)
        if not combos or len(combos['higgsDicts'])==0:
            continue

        if channel=='3lF':
            if events3lF=={}:                                                                            
                events3lF=combos['higgsDicts']
            else:
                for k in events3lF:                                                                                   
                    events3lF[k].extend(combos['higgsDicts'][k])
        else:
            if events=={}:                                                                                               
                events=combos['higgsDicts']                                                                     
            else:                                                                                                   
                for k in events:
                    events[k].extend(combos['higgsDicts'][k])

    dfFlat = pd.DataFrame.from_dict(events)
    dfFlat = shuffle(dfFlat)

    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='2lSS':
        dfFlat.to_csv('csvFiles/higgsTop2lSS/'+outF, index=False, float_format='%.3f')
    else:# channel=='3l':
        dfFlat.to_csv('csvFiles/higgsTop3lS/'+outF, index=False, float_format='%.3f')
        df3lF = pd.DataFrame.from_dict(events3lF)
        df3lF = shuffle(df3lF)
        df3lF.to_csv('csvFiles/higgsTop3lF/'+outF, index=False, float_format='%.3f')

linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=10)(delayed(runReco)(inf) for inf in linelist)

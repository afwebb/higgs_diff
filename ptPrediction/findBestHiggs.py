'''

'''

import pandas as pd
from tensorflow.keras.models import load_model
import ROOT
from ROOT import TFile
from sklearn.utils import shuffle
import numpy as np
import sys
import random
from functionsMatch import jetCombosTop, findBestTopKeras, findBestHiggs
from dictPt import ptDictHiggs2lSS, ptDictHiggs3lF, ptDictHiggs3lS
from joblib import Parallel, delayed
import multiprocessing

#Open input file
#inf = sys.argv[1]
#outDir = sys.argv[2]

def runReco(inf):
    #Set the channel, load in the top model
    if '3l' in inf:
        channel='3l'
        is3l = True
        ptDict = ptDictHiggs3lS

        #topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
        #topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy")

        model3lF = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgs3lF.h5")
        normFactors3lF = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgs3lF_normFactors.npy")

        model3lS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgs3lS.h5")
        normFactors3lS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgs3lS_normFactors.npy")

    elif '2lSS' in inf:
        channel='2lSS'
        ptDict = ptDictHiggs2lSS
        is3l = False

        #topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top2lSS.h5")
        #topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top2lSS_normFactors.npy")
        
        model2lSS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgs2lSS.h5")
        normFactors2lSS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgs2lSS_normFactors.npy")

    else:
        print(f'Channel {channel} is invalid. Should be 2lSS or 3l')
        exit()
        
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    
    #initialize output dicts
    events = []
    events3lF = []
    
    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
            
        nom.GetEntry(idx)

        #Get the Higgs Pt
        for i, pdgId in enumerate(nom.m_truth_pdgId):
            if pdgId ==25:
                higgs_pt = nom.m_truth_pt[i]
                break
        if not higgs_pt:
            continue
            
        #topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
        #topIdx0, topIdx1 = topRes['bestComb']
        #topScore = topRes['topScore']

        isF = False
        if is3l and nom.lep_Parent_0 == 25: 
            isF = True

        if isF:
            res3lF = findBestHiggs(nom, '3lF', model3lF, normFactors3lF)
            higgsScore = res3lF['higgsScore']
            lepIdx = res3lF['bestComb'][0]

            #events3lF.append( ptDictHiggsTop3lF(nom, lepIdx, higgsScore, topIdx0, topIdx1, topScore, higgs_pt) )
            events3lF.append( ptDictHiggs3lF(nom, lepIdx, higgsScore, higgs_pt) )
        else:
            if is3l:
                res = findBestHiggs(nom, '3lS', model3lS, normFactors3lS)
            else:
                res = findBestHiggs(nom, '2lSS', model2lSS, normFactors2lSS)
                
            higgsScore = res['higgsScore']
            lepIdx, jetIdx0, jetIdx1 = res['bestComb']
            events.append( ptDict(nom, jetIdx0, jetIdx1, lepIdx, higgsScore, higgs_pt) )
                                    
            

    dfFlat = pd.DataFrame.from_dict(events)
    dfFlat = shuffle(dfFlat)

    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='2lSS':
        dfFlat.to_csv('inputFiles/higgs2lSS/'+outF, index=False)
    elif channel=='3l':
        dfFlat.to_csv('inputFiles/higgs3lS/'+outF, index=False)
        df3lF = pd.DataFrame.from_dict(events3lF)
        df3lF = shuffle(df3lF)
        df3lF.to_csv('results/higgs3lF/'+outF, index=False)

linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=15)(delayed(runReco)(inf) for inf in linelist)

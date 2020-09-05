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
from functionsMatch import jetCombosTop, findBestTopKeras, findBestHiggsTop
from dictPt import ptDictHiggsTop2lSS, ptDictHiggsTop3lF, ptDictHiggsTop3lS
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
        ptDict = ptDictHiggsTop3lS

        topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy")

        model3lF = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop3lF.h5")
        normFactors3lF = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop3lF_normFactors.npy")

        model3lS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop3lS.h5")
        normFactors3lS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop3lS_normFactors.npy")

    elif '2lSS' in inf:
        channel='2lSS'
        ptDict = ptDictHiggsTop2lSS
        is3l = False

        topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top2lSS.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top2lSS_normFactors.npy")
        
        model2lSS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop2lSS.h5")
        normFactors2lSS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop2lSS_normFactors.npy")

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
            
        topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
        if not topRes: continue
        topIdx0, topIdx1 = topRes['bestComb']
        topScore = topRes['topScore']

        isF = False
        if is3l and nom.lep_Parent_0 == 25: 
            isF = True

        if isF:
            res3lF = findBestHiggsTop(nom, '3lF', model3lF, normFactors3lF, topIdx0, topIdx1, topScore)
            if not res3lF: continue
            higgsTopScore = res3lF['higgsTopScore']
            lepIdx = res3lF['bestComb'][0]

            events3lF.append( ptDictHiggsTop3lF(nom, lepIdx, higgsTopScore, topIdx0, topIdx1, topScore, higgs_pt) )
        else:
            if is3l:
                res = findBestHiggsTop(nom, '3lS', model3lS, normFactors3lS, topIdx0, topIdx1, topScore)
            else:
                res = findBestHiggsTop(nom, '2lSS', model2lSS, normFactors2lSS, topIdx0, topIdx1, topScore)
                
            if not res: continue
            higgsTopScore = res['higgsTopScore']
            lepIdx, jetIdx0, jetIdx1 = res['bestComb']
            events.append( ptDict(nom, jetIdx0, jetIdx1, lepIdx, higgsTopScore, topIdx0, topIdx1, topScore, higgs_pt) )
                                    
            

    dfFlat = pd.DataFrame.from_dict(events)
    dfFlat = shuffle(dfFlat)

    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='2lSS':
        dfFlat.to_csv('inputFiles/higgsTop2lSS/'+outF, index=False)
    elif channel=='3l':
        dfFlat.to_csv('inputFiles/higgsTop3lS/'+outF, index=False)
        df3lF = pd.DataFrame.from_dict(events3lF)
        df3lF = shuffle(df3lF)
        df3lF.to_csv('inputFiles/higgsTop3lF/'+outF, index=False)

linelist = [line.rstrip() for line in open(sys.argv[1])]
#runReco(linelist[0])
Parallel(n_jobs=15)(delayed(runReco)(inf) for inf in linelist)

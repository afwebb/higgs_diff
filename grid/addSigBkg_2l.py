import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math
from rootpy.vector import LorentzVector
from rootpy.io import root_open
from rootpy.tree import Tree, FloatCol, TreeModel
import root_numpy
import sys
import pickle
import xgboost as xgb
import numpy as np

inf = sys.argv[1]
#njet = sys.argv[3]
f = rootpy.io.root_open(inf)
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
oldTree = f.get('nominal')
    #oldTree.SetBranchStatus("*",0)
    #for br in branch_list:
    #    oldTree.SetBranchStatus(br,1)

xgbModelPath = "models/sigBkg/2l_inc.dat"
xgbModel = pickle.load(open(xgbModelPath, "rb"))

events = []

current = 0
for e in oldTree:
    current+=1
    if current%10000==0:
        print(current)

    k = {}

    k['dilep_type'] = e.dilep_type
    k['lep_Pt_0'] = e.lep_Pt_0
    k['lep_Eta_0'] = e.lep_Eta_0
    k['lep_Phi_0'] = e.lep_Phi_0
    k['lep_ID_0'] = e.lep_ID_0
    k['lep_Pt_1'] = e.lep_Pt_1
    k['lep_Eta_1'] = e.lep_Eta_1
    k['lep_Phi_1'] = e.lep_Phi_1
    k['lep_ID_1'] = e.lep_ID_1
    k['Mll01'] = e.Mll01
    k['DRll01'] = e.DRll01
    k['Ptll01'] = e.Ptll01
    k['lead_jetPt'] = e.lead_jetPt 
    k['lead_jetEta'] = e.lead_jetEta
    k['lead_jetPhi'] = e.lead_jetPhi 
    k['sublead_jetPt'] = e.sublead_jetPt 
    k['sublead_jetEta'] = e.sublead_jetEta
    k['sublead_jetPhi'] = e.sublead_jetPhi
    k['HT'] = e.HT 
    k['HT_lep'] = e.HT_lep
    k['nJets_OR_T'] = e.nJets_OR_T
    k['nJets_OR_T_MV2c10_70'] = e.nJets_OR_T_MV2c10_70 
    k['MET_RefFinal_et'] = e.MET_RefFinal_et
    k['MET_RefFinal_phi'] = e.MET_RefFinal_phi
    k['DRlj00'] = e.DRlj00 
    #k['DRlj10'] = e.DRlj10
    k['DRjj01'] = e.DRjj01

    k['dNN_bin_score'] = e.dNN_bin_score_2l
    k['dNN_pt_score'] = e.dNN_pt_score_2l

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)                                                                       
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)

    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
xgbMat = xgb.DMatrix(dFrame, feature_names=list(dFrame))
xgbSigBkg2l = xgbModel.predict(xgbMat)
print(xgbSigBkg2l.shape)

# Decay mode score                                                                                                                              
with root_open(inf, mode='a') as myfile:
    xgbSigBkg2l = np.asarray(xgbSigBkg2l)
    xgbSigBkg2l.dtype = [('xgbSigBkg2l', 'float32')]
    xgbSigBkg2l.dtype.names = ['xgbSigBkg2l']
    root_numpy.array2tree(xgbSigBkg2l, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

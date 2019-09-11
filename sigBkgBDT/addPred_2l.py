import xgboost as xgb
import ROOT
from ROOT import TFile
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
from rootpy.io import root_open
from rootpy.tree import Tree, FloatCol, TreeModel
import root_numpy
import sys
import pickle
import math
import numpy as np
from dict_top import topDict
from dict_higgs import higgsDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
#from rootpy.io import root_open

#Read in list of files
inf = sys.argv[1]

#load xgb models

modelPath = "xgb_models/2l/xgb_match_higgsLepCut.dat"
model = pickle.load(open(modelPath, "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

#create pt prediction dicts
def create_dict(nom):
    current = 0

    events = []
    
    nEntries = nom.GetEntries()
    print(nEntries)
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))

        nom.GetEntry(idx)

        k = {}
        k['dilep_type'] = nom.dilep_type
        k['lep_Pt_0'] = nom.lep_Pt_0
        k['lep_Eta_0'] = nom.lep_Eta_0
        k['lep_Phi_0'] = nom.lep_Phi_0
        k['lep_ID_0'] = nom.lep_ID_0
        k['lep_Pt_1'] = nom.lep_Pt_1
        k['lep_Eta_1'] = nom.lep_Eta_1
        k['lep_Phi_1'] = nom.lep_Phi_1
        k['lep_ID_1'] = nom.lep_ID_1
        k['Mll01'] = nom.Mll01
        k['DRll01'] = nom.DRll01
        k['Ptll01'] = nom.Ptll01
        k['lead_jetPt'] = nom.lead_jetPt
        k['lead_jetEta'] = nom.lead_jetEta
        k['lead_jetPhi'] = nom.lead_jetPhi
        k['sublead_jetPt'] = nom.sublead_jetPt
        k['sublead_jetEta'] = nom.sublead_jetEta
        k['sublead_jetPhi'] = nom.sublead_jetPhi
        k['HT'] = nom.HT
        k['HT_lep'] = nom.HT_lep
        k['nJets_OR_T'] = e.nJets_OR_T
        k['nJets_OR_T_MV2c10_70'] = e.nJets_OR_T_MV2c10_70
        k['MET_RefFinal_et'] = e.MET_RefFinal_et
        k['MET_RefFinal_phi'] = e.MET_RefFinal_phi
        k['DRlj00'] = e.DRlj00
        k['DRjj01'] = e.DRjj01
        
        k['bin_score'] = e.xgb_bin_score_2l
        k['pt_score'] = e.xgb_pt_score_2l
        k['higgsScore'] = e.higgsScore
        k['topScore'] = e.topScore
        
        events.append(k)

    return events

#loop over file list, add prediction branches
def run_pred(inputPath):
    print(inputPath)
    f = TFile(inputPath, "READ")

    dsid = inputPath.split('/')[-1]
    dsid = dsid.replace('.root', '')
    print(dsid)
    
    nom = f.Get('nominal')
    if nom.GetEntries() == 0:
        return 0
    
    event_dict = create_dict(nom)
    inDF = pd.DataFrame(event_dict)
    xgbMat = xgb.DMatrix(inDF, feature_names=list(inDF))
    y_pred = model.predict(xgbMat)
    
    with root_open(inputPath, mode='a') as myfile:
        xgbScore_sigBkg_2l = np.asarray(y_pred)
        xgbScore_sigBkg_2l.dtype = [('xgbScore_sigBkg_2l', 'float32')]
        xgbScore_sigBkg_2l.dtype.names = ['xgbScore_sigBkg_2l']
        root_numpy.array2tree(xgbScore_sigBkg_2l, tree=myfile.nominal)
        
        myfile.write()
        myfile.Close()

linelist = [line.rstrip() for line in open(inf)]
print(linelist)
Parallel(n_jobs=10)(delayed(run_pred)(inFile) for inFile in linelist)


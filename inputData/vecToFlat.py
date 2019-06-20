#Script to convert 4-vec dataframe to a flat dataframe
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
import sys
import torch
from rootpy.vector import LorentzVector

inFile = sys.argv[1]
outFile = sys.argv[2]

inDF = pd.read_csv(inFile)

#Branches: MET,MET_phi,comboScore,higgs_pt,jet_E_h0,jet_E_h1,jet_Eta_h0,jet_Eta_h1,jet_MV2c10_h0,jet_MV2c10_h1,jet_Phi_h0,jet_Phi_h1,jet_Pt_h0,jet_Pt_h1,lep_E_H,lep_E_O,lep_Eta_H,lep_Eta_O,lep_Phi_O,lep_Pt_H,lep_Pt_O,top_E_0,top_E_1,top_Eta_0,top_Eta_1,top_MV2c10_0,top_MV2c10_1,top_Phi_0,top_Phi_1,top_Pt_0,top_Pt_1

events = []

current = 0
total = inDF.shape[0]

for index, k in inDF.iterrows():

    if current%10000==0:
        print(str(current)+'/'+str(total))
    current+=1

    met = LorentzVector()
    met.SetPtEtaPhiE(k['MET'], 0, k['MET_phi'], k['MET'])

    lepH = LorentzVector()
    lepH.SetPtEtaPhiE(k['lep_Pt_H'], k['lep_Eta_H'], 0, k['lep_E_H'])

    lepO = LorentzVector()
    lepO.SetPtEtaPhiE(k['lep_Pt_O'], k['lep_Eta_O'], k['lep_Phi_O'], k['lep_E_O'])

    jet0 = LorentzVector()
    jet0.SetPtEtaPhiE(k['jet_Pt_h0'], k['jet_Eta_h0'], k['jet_Phi_h0'], k['jet_E_h0'])

    jet1 = LorentzVector()
    jet1.SetPtEtaPhiE(k['jet_Pt_h1'], k['jet_Eta_h1'], k['jet_Phi_h1'], k['jet_E_h1'])

    top0 = LorentzVector()
    top0.SetPtEtaPhiE(k['top_Pt_0'], k['top_Eta_0'], k['top_Phi_0'], k['top_E_0'])

    top1 = LorentzVector()
    top1.SetPtEtaPhiE(k['top_Pt_1'], k['top_Eta_1'], k['top_Phi_1'], k['top_E_1'])
    
    q = {}
    q['higgs_pt'] = k['higgs_pt']

    q['comboScore'] = k['comboScore']

    q['nJets'] = k['nJets'] 
    q['nJets_MV2c10_70'] = k['nJets_MV2c10_70']

    q['jet_MV2c10_h0'] = k['jet_MV2c10_h0']
    q['jet_MV2c10_h1'] = k['jet_MV2c10_h1']
    
    q['jet_Pt_0'] = k['jet_Pt_h0']
    q['jet_Pt_1'] = k['jet_Pt_h1']
    
    q['lep_Pt_H'] = k['lep_Pt_H']
    q['lep_Pt_O'] = k['lep_Pt_O']

    q['top_Pt_0'] = k['top_Pt_0']
    q['top_Pt_1'] = k['top_Pt_1']

    q['M(jjl)'] = (jet0+jet1+lepH).M()
    q['Pt(jjl)'] = (jet0+jet1+lepH).Pt()

    q['Mjj'] = (jet0+jet1).M()
    q['dR(jj, l)'] = (jet0+jet1).DeltaR(lepH)

    q['dRjl0H'] = jet0.DeltaR(lepH)
    q['dRjl1H'] = jet1.DeltaR(lepH)

    q['dRjl0O'] = jet0.DeltaR(lepO)
    q['dRjl1O'] = jet1.DeltaR(lepO)
    q['dRll'] = lepH.DeltaR(lepO)

    q['dRtl0H'] = top0.DeltaR(lepH)
    q['dRtl1H'] = top1.DeltaR(lepH)

    q['dRtj00'] = top0.DeltaR(jet0)
    q['dRtj10'] = top1.DeltaR(jet0)

    q['dRtj01'] = top0.DeltaR(jet1)
    q['dRtj11'] = top1.DeltaR(jet1)

    events.append(q)

dFrame = pd.DataFrame(events)
dFrame.to_csv(outFile, index=False)

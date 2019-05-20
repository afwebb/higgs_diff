#Script to test whether the combination of leptons/jets with the highest xgb score corresponds to the correct combo

import ROOT
import numpy as np
import uproot
import sys
import math
import pickle
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
#import xgboost as xgb
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from dict_top import flatDict

inputFile = sys.argv[1]
#modelPath = sys.argv[2]
#topModelPath = sys.argv[2]
f=uproot.open(inputFile)
nom=f.get('nominal')

totalPast = 0
nCorrect = 0
lepCorrect = 0
lep1jCorrect = 0

events = []
bestScores = []

truthScores = []
badScores = []

wrongTruth = []
wrongPred = []
rightTruth = []

#xgbModel = pickle.load(open(modelPath, "rb"))
#topModel = pickle.load(open(topModelPath, "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi


la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*'])
'''
from collections import namedtuple
parttype = namedtuple('parttype', ['barcode', 'pdgid', 'status', 'eta', 'phi', 'pt', 'parents', 'children'])

def make_partdict(la, idx):
    rv = dict(zip(la[b'm_truth_barcode'][idx],
                 (parttype(*_2) for
                  _2 in zip(la[b'm_truth_barcode'][idx],
                            la[b'm_truth_pdgId'][idx],
                            la[b'm_truth_status'][idx],
                            la[b'm_truth_eta'][idx],
                            la[b'm_truth_phi'][idx],
                            la[b'm_truth_pt'][idx],
                            la[b'm_truth_parents'][idx],
                            la[b'm_truth_children'][idx])
                 )
                 ))
    return rv

'''

class Net(nn.Module):

    def __init__(self, D_in, nodes, layers):
        self.layers = layers
        super().__init__()
        self.fc1 = nn.Linear(D_in, nodes)
        self.relu1 = nn.LeakyReLU()
        self.dout = nn.Dropout(0.2)
        #self.fc2 = nn.Linear(50, 100)                                                                                           \
                                                                                                                                  
        self.fc = nn.Linear(nodes, nodes)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(nodes, 2)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        h1 = self.dout(self.relu1(self.fc1(input_)))
        for i in range(self.layers):
            h1 = self.dout(self.relu1(self.fc(h1)))
        a1 = self.out(h1)
        y = self.out_act(a1)
        return y

net = Net(19,50,4)
net.load_state_dict(torch.load('../truthMatching/models/model_topAllBad_4l_50n.pt'))
net.eval()
net.train(False)

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

def drCheck(eta, phi, truth_eta, truth_phi, cut):
    dr = sqrt( (phi-truth_phi)**2 + (eta-truth_eta)**2 )
  
    return dr < cut

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
nMatch = 0
higgVecs = []

truthDicts = []

fourVecDicts = []

for idx in range(len(la[b'met']) ):
    #current+=1
    if idx%10000==0:
        print(idx)                                                                                                 
    if idx==1000:
        break

    if len(la[b'lep_pt'][idx])!=2: continue
        
    truthComb = []

    higgCand = LorentzVector()
    
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])
    
    fourVecs['met'] = met
    
    higgCand+=met
        
    lep4Vecs = []

    lepMatch = -1
    for i in range(2):

        lep_pt = la[b'lep_pt'][idx][i]
        lep_eta = la[b'lep_eta'][idx][i]
        lep_phi = la[b'lep_phi'][idx][i]
        lep_E = la[b'lep_E'][idx][i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        lep4Vecs.append(lepVec)

        if la[b'lep_parent'][idx][i]==25:
            lepMatch=i

    if lepMatch == -1:
        continue

    truthComb.append(lepMatch)

    match=0  
    truthJets = []

    higgsJets=[]
    higgsJetsMV2c10=[]

    badJets=[]
    badJetsMV2c10=[]

    jet4Vecs = []
    jet4VecsMV2c10 = []

    match = 0

    for i in range(len(la[b'jet_pt'][idx])):#la[b'selected_jets'][i]:

        jet_pt = la[b'jet_pt'][idx][i]
        jet_eta = la[b'jet_eta'][idx][i]
        jet_phi = la[b'jet_phi'][idx][i]
        jet_E = la[b'jet_E'][idx][i]
        jet_MV2c10 = la[b'jet_MV2c10'][idx][i]
        
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)
        
        jet4Vecs.append(jetVec)
        jet4VecsMV2c10.append(jet_MV2c10)
        
        if abs(la[b'jet_parent'][idx][i])==6:
            truthJets.append(i)
            higgCand+=jetVec
            higgsJets.append(jetVec)
            higgsJetsMV2c10.append(jet_MV2c10)
            match+=1
            truthComb.append(i)
        else:
            badJets.append(jetVec)
            badJetsMV2c10.append(jet_MV2c10)
               
    if match!=2: continue

    combosTop = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]

                #k = flatDict( lep4Vecs[l], jet4Vecs[i], jet4Vecs[j], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j] )                                   
                t = flatDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j] )

                combosTop.append([t, comb])

    #loop over combinations, score them in the BDT, figure out the best result                                                    
    topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
    topMat = torch.tensor(topDF.values, dtype=torch.float32)

    topMat = normalize(topMat)

    topPred = net(topMat)[:,1].detach().numpy()
    topBest = np.argmax(topPred)

    bestTopComb = combosTop[topBest][1]
    topMatches = bestTopComb[1:]

    for c, s in zip(combosTop, topPred):
        if c[1][0]==truthComb[0] and c[1][1] in truthComb[1:] and c[1][2] in truthComb[1:]:
            #truthK = {**c[0], **c[1]}
            #print('truth score', s)
            truthScores.append(s)
        else:
            badScores.append(s)
            
    #predK = {**combos[best][0], **combos[best][1]}

    #print(sorted(bestTopComb[1:]), sorted(truthComb[1:]), topPred[topBest])
    #print(combosTop[topBest][0])

    totalPast+=1

    if sorted(bestTopComb[1:])==sorted(truthComb[1:]):
        nCorrect+=1
        #rightTruth.append(truthK)
    #else:
    #    wrongTruth.append(truthK)
    #    wrongPred.append(predK)

    if bestTopComb[1] in truthComb[1:] or bestTopComb[2] in truthComb[1:]:
        lep1jCorrect+=1

print('events passed', totalPast)
print('percent correct', nCorrect/totalPast)
#print('lep correct', lepCorrect/totalPast)
print('lep and 1 jet correct', lep1jCorrect/totalPast)
'''
rightTruthDF = pd.DataFrame(rightTruth)
wrongTruthDF = pd.DataFrame(wrongTruth)
wrongPredDF = pd.DataFrame(wrongPred)

rightTruthDF.to_csv('outputData/rightTruthAllBad.csv', index=False)
wrongTruthDF.to_csv('outputData/wrongTruthAllBad.csv', index=False)
wrongPredDF.to_csv('outputData/wrongPredAllBad.csv', index=False)


trDF = pd.DataFrame.from_dict(truthDicts)                                                               
trMat = torch.tensor(trDF.values, dtype=torch.float32)                                                  
trMat = normalize(trMat)                                                                                                                     
print(trDF)                                                                                                                                  

truthScores = net(trMat)[:,1].detach().numpy()                                                                                                 
#print('good', net(trMat), truthComb)                                                                         '''                                

plt.figure()
plt.hist(truthScores)
plt.savefig('plots/topTruthScores.png')

plt.figure()
plt.hist(badScores)
plt.savefig('plots/topBadScores.png')

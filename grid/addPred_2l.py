import ROOT
import pandas as pd
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
from rootpy.io import root_open
from rootpy.tree import Tree, FloatCol, TreeModel
import root_numpy
import sys
import pickle
import math
import xgboost as xgb
import numpy as np
from dict_top import topDict
from dict_higgs import higgsDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
#from rootpy.io import root_open

#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

inf = sys.argv[1]
#outFile = sys.argv[2]
higgsModelPath = "models/2l/xgb_match_higgsLepCut.dat"
topModelPath = "models/2l/xgb_match_topLepCut.dat"
#outFile = sys.argv[4]

#njet = sys.argv[3]
f = ROOT.TFile(inf, "READ")

dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)

nom = f.Get('nominal')
    #oldTree.SetBranchStatus("*",0)
    #for br in branch_list:
    #    oldTree.SetBranchStatus(br,1)

events = []
bestScores = []

xgbModel = pickle.load(open(higgsModelPath, "rb"))

topModel = pickle.load(open(topModelPath, "rb"))

normFactors = np.load('models/2l/normFactors.npy')
normFactors = torch.from_numpy(normFactors).float()
yMax = normFactors[0]#torch.from_numpy(normFactors[0]).float()#Y.max(0, keepdim=True)[0]
xMax = normFactors[1:]#torch.from_numpy(normFactors[1:]).float()#X.max(0, keepdim=True)[0]

#xgbPtModel = pickle.load(open('models/2l/xgb_higgsPflowFlatBranch.dat

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

def create_dict():
    current = 0
    
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))

        nom.GetEntry(idx)

        #if nom.total_leptons < 2: continue
        #if nom.dilep_type < 1: continue                                                                                                
        #if nom.total_charge == 0: continue  
        #if nom.nJets_OR_T_MV2c10_70 < 1: continue
        #if nom.nJets_OR_T < 2: continue

        higgCand = LorentzVector()
        
        lep4Vecs = []
        jet4Vecs = []
        
        btags = []
        
        met = LorentzVector()
        met.SetPtEtaPhiE(nom.MET_RefFinal_et, 0, nom.MET_RefFinal_phi, nom.MET_RefFinal_et)

        #for i in range(2):
        lep_Pt_0 = nom.lep_Pt_0
        lep_Eta_0 = nom.lep_Eta_0
        lep_Phi_0 = nom.lep_Phi_0
        lep_E_0 = nom.lep_E_0
        
        lepVec_0 = LorentzVector()
        lepVec_0.SetPtEtaPhiE(lep_Pt_0, lep_Eta_0, lep_Phi_0, lep_E_0)
        lep4Vecs.append(lepVec_0)

        lep_Pt_1 = nom.lep_Pt_1
        lep_Eta_1 = nom.lep_Eta_1
        lep_Phi_1 = nom.lep_Phi_1
        lep_E_1 = nom.lep_E_1

        lepVec_1 = LorentzVector()
        lepVec_1.SetPtEtaPhiE(lep_Pt_1, lep_Eta_1, lep_Phi_1, lep_E_1)
        lep4Vecs.append(lepVec_1)

        for j in range(len(nom.m_pflow_jet_pt)):#nom.selected_jets'][i]:
            jetVec = LorentzVector()
            jetVec.SetPtEtaPhiM(nom.m_pflow_jet_pt[j], nom.m_pflow_jet_eta[j], nom.m_pflow_jet_phi[j], nom.m_pflow_jet_m[j])
            jet4Vecs.append(jetVec)
            
            btags.append(nom.m_pflow_jet_flavor_weight_MV2c10[j])

        combos = []
        combosTop = []
        
        for l in range(len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]
                
                    t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], met, btags[i], btags[j],
                                 nom.m_pflow_jet_jvt[i], nom.m_pflow_jet_jvt[j],
                                 nom.m_pflow_jet_numTrk[i], nom.m_pflow_jet_numTrk[j] )
                    
                    combosTop.append([t, comb])

        #loop over combinations, score them in the BDT, figure out the best result
        topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
        topMat = xgb.DMatrix(topDF, feature_names=list(topDF))
        
        topPred = topModel.predict(topMat)
        topBest = np.argmax(topPred)
        
        bestTopComb = combosTop[topBest][1]
        topMatches = bestTopComb[1:]

        for l in range(len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]

                    if l==0:
                        k = higgsDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, btags[i], btags[j], lep4Vecs[1],
                                       nom.m_pflow_jet_jvt[i], nom.m_pflow_jet_jvt[j],
                                       nom.m_pflow_jet_numTrk[i], nom.m_pflow_jet_numTrk[j])
                    else:
                        k = higgsDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, btags[i], btags[j], lep4Vecs[0],
                                       nom.m_pflow_jet_jvt[i], nom.m_pflow_jet_jvt[j],
                                       nom.m_pflow_jet_numTrk[i], nom.m_pflow_jet_numTrk[j])

                    combos.append([k, comb])


        ###Evaluate higgsTop BDT
        df = pd.DataFrame.from_dict([x[0] for x in combos])
        xgbMat = xgb.DMatrix(df, feature_names=list(df))
        
        pred = xgbModel.predict(xgbMat)
        best = np.argmax(pred)
        
        bestScores.append(pred[best])
        
        bestComb = combos[best][1]
        lepMatch = bestComb[0]
        jetMatches = bestComb[1:]
        
        k = {}
        #k['higgs_pt'] = nom.higgs_pt
        k['comboScore'] = pred[best]
        k['topScore'] = topPred[topBest]
        
        if lepMatch == 0:
            k['lep_Pt_H'] = nom.lep_Pt_0
            k['lep_Eta_H'] = nom.lep_Eta_0
            phi_0 = nom.lep_Phi_0
            k['lep_E_H'] = nom.lep_E_0
            
            k['lep_Pt_O'] = nom.lep_Pt_1
            k['lep_Eta_O'] = nom.lep_Eta_1
            k['lep_Phi_O'] = calc_phi(phi_0, nom.lep_Phi_1)
            k['lep_E_O'] = nom.lep_E_1

        elif lepMatch == 1:
            k['lep_Pt_H'] = nom.lep_Pt_1
            k['lep_Eta_H'] = nom.lep_Eta_1
            phi_0 = nom.lep_Phi_1
            k['lep_E_H'] = nom.lep_E_1

            k['lep_Pt_O'] = nom.lep_Pt_0
            k['lep_Eta_O'] = nom.lep_Eta_0
            k['lep_Phi_O'] = calc_phi(phi_0, nom.lep_Phi_0)
            k['lep_E_O'] = nom.lep_E_0

            
        n = 0
        for i in jetMatches:#nom.nJets_OR_T):
            
            k['jet_Pt_h'+str(n)] = nom.m_pflow_jet_pt[i]
            k['jet_Eta_h'+str(n)] = nom.m_pflow_jet_eta[i]
            k['jet_E_h'+str(n)] = jet4Vecs[i].E()#nom.m_pflow_jet_E[i]
            k['jet_Phi_h'+str(n)] = calc_phi(phi_0, nom.m_pflow_jet_phi[i])
            k['jet_MV2c10_h'+str(n)] = nom.m_pflow_jet_flavor_weight_MV2c10[i]
            
            n+=1

        btags = np.array(btags)
            
        btags[jetMatches[0]] = 0
        btags[jetMatches[1]] = 0
        bestBtags = np.argpartition(btags, -2)[-2:]
        
        n = 0
        for i in topMatches:#bestBtags:#nom.nJets_OR_T):      
            k['top_Pt_'+str(n)] = nom.m_pflow_jet_pt[i]
            k['top_Eta_'+str(n)] = nom.m_pflow_jet_eta[i]
            k['top_E_'+str(n)] = jet4Vecs[i].E()#nom.m_pflow_jet_E[i]
            k['top_Phi_'+str(n)] = calc_phi(phi_0, nom.m_pflow_jet_phi[i])
            k['top_MV2c10_'+str(n)] = nom.m_pflow_jet_flavor_weight_MV2c10[i]
            
            n+=1

        k['MET'] = nom.MET_RefFinal_et
        k['MET_phi'] = calc_phi(phi_0, nom.MET_RefFinal_phi)

        events.append(k)

    return events

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
        self.out = nn.Linear(nodes, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        h1 = self.dout(self.relu1(self.fc1(input_)))
        for i in range(self.layers):
            h1 = self.dout(self.relu1(self.fc(h1)))
        a1 = self.out(h1)
        y = self.out_act(a1)
        return y


def make_tensors(inDF, xMax):
    
    #inDF['higgs_pt'] = pd.cut(inDF['higgs_pt'], bins=[0, 150000, 9999999999], labels=[0,1])
    #Y = inDF['higgs_pt']
    X = inDF #.drop(['higgs_pt'],axis=1)
    
    X = torch.tensor(X.values, dtype=torch.float32)
    #Y = torch.FloatTensor(Y.values)
    
    #yMax = torch.from_numpy(normFactors[0]).float()#Y.max(0, keepdim=True)[0]
    #xMax = torch.from_numpy(normFactors[1:]).float()#X.max(0, keepdim=True)[0]
    
    X = X / xMax
    #Y = Y / yMax
    
    return X #, Y


def pred_pt(X, yMax):

    #print(X.shape[1])
    net_pt = Net(X.size()[1],90,6)
    net_pt.load_state_dict(torch.load('models/2l/torch_pt_6l_90n.pt'))
    net_pt.eval()

    Y_pred_pt = net_pt(X)[:,0]

    return (Y_pred_pt*yMax).float().detach().numpy()


def pred_bin(X):

    net_bin = Net(X.size()[1], 125, 5)
    net_bin.load_state_dict(torch.load('models/2l/torch_binned_5l_125n.pt'))
    net_bin.eval()
    
    Y_pred_bin = net_bin(X)[:,0]

    return Y_pred_bin.float().detach().numpy()


event_dict = create_dict()

inDF = pd.DataFrame(event_dict)

X = make_tensors(inDF, xMax)

y_pred_pt = pred_pt(X, yMax)
y_pred_bin = pred_bin(X)

inDF['y_pred_pt'] = y_pred_pt
inDF['y_pred_bin'] = y_pred_bin

#inDF.to_csv('test.csv', index=False)

#f.Close()

#y_pred_pt = y_pred_pt.float().detach().numpy()
#y_pred_bin = y_pred_bin.float().detach().numpy()

with root_open(inf, mode='a') as myfile:
    dNN_pt_score = np.asarray(y_pred_pt)
    dNN_pt_score.dtype = [('dNN_pt_score_2l', 'float32')]
    dNN_pt_score.dtype.names = ['dNN_pt_score_2l']
    root_numpy.array2tree(dNN_pt_score, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

with root_open(inf, mode='a') as myfile:
    dNN_bin_score = np.asarray(y_pred_bin)
    dNN_bin_score.dtype = [('dNN_bin_score_2l', 'float32')]
    dNN_bin_score.dtype.names = ['dNN_bin_score_2l']
    root_numpy.array2tree(dNN_bin_score, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

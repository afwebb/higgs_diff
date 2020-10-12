'''                                                                                                                           Add the regressed higgs pt to a ROOT file, as well as the output of intermediate models. Takes a list of ROOT files as inputs\, generate a new ROOT file with predictions as output. Runs over each input in parallel                                       Usage: python parallelPred_2l.py <input files>                                                                                '''

import ROOT
from ROOT import TFile
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
from dict_3lDecay import decayDict
from dict_top3l import topDict
from dict_higgs1l import higgs1lDict
from dict_higgs2l import higgs2lDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing

inf = sys.argv[1]

#Load XGB models
decayModelPath = "models/3l/xgb_decay_fullLep.dat"
higgs1lModelPath = "models/3l/xgb_match_higgs1lLepCut.dat"
higgs2lModelPath = "models/3l/xgb_match_higgs2lLepCut.dat"
topModelPath = "models/3l/xgb_match_top3lLepCut.dat"

decayModel = pickle.load(open(decayModelPath, "rb"))

higgs1lModel = pickle.load(open(higgs1lModelPath, "rb"))
higgs2lModel = pickle.load(open(higgs2lModelPath, "rb"))

topModel = pickle.load(open(topModelPath, "rb"))

normFactors1l = np.load('models/3l/normFactors_1l.npy')
normFactors1l = torch.from_numpy(normFactors1l).float()
yMax1l = normFactors1l[0]#torch.from_numpy(normFactors[0]).float()#Y.max(0, keepdim=True)[0]
xMax1l = normFactors1l[1:]#torch.from_numpy(normFactors[1:]).float()#X.max(0, keepdim=True)[0]

normFactors2l = np.load('models/3l/normFactors_2l.npy')
normFactors2l = torch.from_numpy(normFactors2l).float()
yMax2l = normFactors2l[0]#torch.from_numpy(normFactors[0]).float()#Y.max(0, keepdim=True)[0]
xMax2l = normFactors2l[1:]

#Load torch models
class Net(nn.Module):

    def __init__(self, D_in, nodes, layers):
        self.layers = layers
        super().__init__()
        self.fc1 = nn.Linear(D_in, nodes)
        self.relu1 = nn.LeakyReLU()
        self.dout = nn.Dropout(0.2)

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


net_bin1l = Net(34, 125, 5)
net_bin1l.load_state_dict(torch.load('models/3l/model_higgs1lBin_5l_125n.pt'))
net_bin1l.eval()

net_bin2l = Net(24, 125, 5)
net_bin2l.load_state_dict(torch.load('models/3l/model_higgs2lBin_5l_125n.pt'))
net_bin2l.eval()

net_pt1l = Net(34, 90, 6)
net_pt1l.load_state_dict(torch.load('models/3l/model_higgs1l_6l_90n.pt'))
net_pt1l.eval()

net_pt2l = Net(24,90,6)
net_pt2l.load_state_dict(torch.load('models/3l/model_higgs2l_6l_90n.pt'))
net_pt2l.eval()

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

def create_dict(nom):
    current = 0

    events1l = []
    events2l = []
    decayDicts = []
    bestScores = []

    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
            #if current%100000==0:
            #break
        nom.GetEntry(idx)

        higgCand = LorentzVector()
        
        lep4Vecs = []
        jet4Vecs = []
        
        btags = []
        
        met = LorentzVector()
        met.SetPtEtaPhiE(nom.MET_RefFinal_et, 0, nom.MET_RefFinal_phi, nom.MET_RefFinal_et)

        lepVec_0 = LorentzVector()
        lepVec_0.SetPtEtaPhiE(nom.lep_Pt_0, nom.lep_Eta_0, nom.lep_Phi_0, nom.lep_E_0)
        lep4Vecs.append(lepVec_0)

        lepVec_1 = LorentzVector()
        lepVec_1.SetPtEtaPhiE(nom.lep_Pt_1, nom.lep_Eta_1, nom.lep_Phi_1, nom.lep_E_1)
        lep4Vecs.append(lepVec_1)

        lepVec_2 = LorentzVector()
        lepVec_2.SetPtEtaPhiE(nom.lep_Pt_2, nom.lep_Eta_2, nom.lep_Phi_2, nom.lep_E_2)
        lep4Vecs.append(lepVec_2)

        for j in range(len(nom.m_pflow_jet_pt)):#nom.selected_jets'][i]:
            jetVec = LorentzVector()
            jetVec.SetPtEtaPhiM(nom.m_pflow_jet_pt[j], nom.m_pflow_jet_eta[j], nom.m_pflow_jet_phi[j], nom.m_pflow_jet_m[j])
            jet4Vecs.append(jetVec)
            
            btags.append(nom.m_pflow_jet_flavor_weight_MV2c10[j])

        combosTop = []
        
        for l in range(len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]
                
                    t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met, btags[i], btags[j],
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

        combos1l = []

        for l in range(1, len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]
                    if l==1:
                        k = higgs1lDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, 
                                         nom.m_pflow_jet_flavor_weight_MV2c10[i], nom.m_pflow_jet_flavor_weight_MV2c10[j], 
                                         lep4Vecs[0], lep4Vecs[2],
                                         nom.m_pflow_jet_jvt[i], nom.m_pflow_jet_jvt[j],
                                         nom.m_pflow_jet_numTrk[i], nom.m_pflow_jet_numTrk[j])
                    else:
                        k = higgs1lDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, 
                                         nom.m_pflow_jet_flavor_weight_MV2c10[i], nom.m_pflow_jet_flavor_weight_MV2c10[j], 
                                         lep4Vecs[0], lep4Vecs[1],
                                         nom.m_pflow_jet_jvt[i], nom.m_pflow_jet_jvt[j],
                                         nom.m_pflow_jet_numTrk[i], nom.m_pflow_jet_numTrk[j])
                        
                    combos1l.append([k, comb])

        combos2l = []
        
        possCombs = [[0,1,2],[0,2,1]]
        for comb in possCombs:
            k = higgs2lDict( lep4Vecs[ comb[0] ], lep4Vecs[ comb[1] ], lep4Vecs[ comb[2] ], met)
            combos2l.append([k, [comb[0], comb[1]] ])


        #Run 2l XGB, find best match                                                                                                                 
        df2l = pd.DataFrame.from_dict([x[0] for x in combos2l])
        xgbMat2l = xgb.DMatrix(df2l, feature_names=list(df2l))
        
        pred2l = higgs2lModel.predict(xgbMat2l)
        best2l = np.argmax(pred2l)
        
        bestComb2l = combos2l[best2l][1]
        lepMatch2l = bestComb2l[1]

        #Run 1l XGB, find best match                                                                                                                 
        df1l = pd.DataFrame.from_dict([x[0] for x in combos1l])
        xgbMat1l = xgb.DMatrix(df1l, feature_names=list(df1l))
        
        pred1l = higgs1lModel.predict(xgbMat1l)
        best1l = np.argmax(pred1l)

        bestComb1l = combos1l[best1l][1]
        lepMatch1l = bestComb1l[0]
        jetMatches1l = bestComb1l[1:]

        ### Add decay dict
        
        k = decayDict( lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met, jet4Vecs[ topMatches[0] ], jet4Vecs[ topMatches[1] ] )
        k['nJets'] = nom.nJets_OR_T
        k['nJets_MV2c10_70'] = nom.nJets_OR_T_MV2c10_70
        k['higgs2l_score'] = pred2l[best2l]
        k['higgs1l_score'] = pred1l[best1l]
        decayDicts.append(k)


        ### Add 2l pt prediction dict

        q = {}
        q['comboScore'] = pred2l[best2l]
        
        if lepMatch2l == 1:
            q['lep_Pt_0'] = nom.lep_Pt_0
            q['lep_Eta_0'] = nom.lep_Eta_0
            phi_0 = nom.lep_Phi_0
            q['lep_E_0'] = nom.lep_E_0

            q['lep_Pt_1'] = nom.lep_Pt_1
            q['lep_Eta_1'] = nom.lep_Eta_1
            q['lep_Phi_1'] = calc_phi(phi_0, nom.lep_Phi_1)
            q['lep_E_1'] = nom.lep_E_1
            
            q['lep_Pt_2'] = nom.lep_Pt_2
            q['lep_Eta_2'] = nom.lep_Eta_2
            q['lep_Phi_2'] = calc_phi(phi_0, nom.lep_Phi_2)
            q['lep_E_2'] = nom.lep_E_2
            
        elif lepMatch2l == 2:
            q['lep_Pt_0'] = nom.lep_Pt_0
            q['lep_Eta_0'] = nom.lep_Eta_0
            phi_0 = nom.lep_Phi_0
            q['lep_E_0'] = nom.lep_E_0
            
            q['lep_Pt_1'] = nom.lep_Pt_2
            q['lep_Eta_1'] = nom.lep_Eta_2
            q['lep_Phi_1'] = calc_phi(phi_0, nom.lep_Phi_2)
            q['lep_E_1'] = nom.lep_E_2
            
            q['lep_Pt_2'] = nom.lep_Pt_1
            q['lep_Eta_2'] = nom.lep_Eta_1
            q['lep_Phi_2'] = calc_phi(phi_0, nom.lep_Phi_1)
            q['lep_E_2'] = nom.lep_E_1

        n = 0
        for i in topMatches:
            q['top_Pt_'+str(n)] = nom.m_pflow_jet_pt[i]
            q['top_Eta_'+str(n)] = nom.m_pflow_jet_eta[i]
            q['top_E_'+str(n)] = jet4Vecs[i].E()#nom.m_pflow_jet_E[i]
            q['top_Phi_'+str(n)] = calc_phi(phi_0, nom.m_pflow_jet_phi[i])
            q['top_MV2c10_'+str(n)] = nom.m_pflow_jet_flavor_weight_MV2c10[i]

            n+=1

        q['MET'] = nom.MET_RefFinal_et
        q['MET_phi'] = calc_phi(phi_0, nom.MET_RefFinal_phi)

        events2l.append(q)

        ### Add 1l Pt prediction dict

        y = {}
        #y['higgs_pt'] = nom.higgs_pt
        y['comboScore'] = pred1l[best1l]
        
        if lepMatch1l == 1:
            y['lep_Pt_H'] = nom.lep_Pt_1
            y['lep_Eta_H'] = nom.lep_Eta_1
            phi_0 = nom.lep_Phi_1
            y['lep_E_H'] = nom.lep_E_1
            
            y['lep_Pt_0'] = nom.lep_Pt_0
            y['lep_Eta_0'] = nom.lep_Eta_0
            y['lep_Phi_0'] = calc_phi(phi_0, nom.lep_Phi_0)
            y['lep_E_0'] = nom.lep_E_0
            
            y['lep_Pt_1'] = nom.lep_Pt_2
            y['lep_Eta_1'] = nom.lep_Eta_2
            y['lep_Phi_1'] = calc_phi(phi_0, nom.lep_Phi_2)
            y['lep_E_1'] = nom.lep_E_2

        elif lepMatch1l == 2:
            y['lep_Pt_H'] = nom.lep_Pt_2
            y['lep_Eta_H'] = nom.lep_Eta_2
            phi_0 = nom.lep_Phi_2
            y['lep_E_H'] = nom.lep_E_2
            
            y['lep_Pt_0'] = nom.lep_Pt_0
            y['lep_Eta_0'] = nom.lep_Eta_0
            y['lep_Phi_0'] = calc_phi(phi_0, nom.lep_Phi_0)
            y['lep_E_0'] = nom.lep_E_0
            
            y['lep_Pt_1'] = nom.lep_Pt_1
            y['lep_Eta_1'] = nom.lep_Eta_1
            y['lep_Phi_1'] = calc_phi(phi_0, nom.lep_Phi_1)
            y['lep_E_1'] = nom.lep_E_1
            
        n = 0
        for i in jetMatches1l:#nom.nJets_OR_T):
            
            y['jet_Pt_h'+str(n)] = nom.m_pflow_jet_pt[i]
            y['jet_Eta_h'+str(n)] = nom.m_pflow_jet_eta[i]
            y['jet_E_h'+str(n)] = jet4Vecs[i].E()#nom.m_pflow_jet_E[i]
            y['jet_Phi_h'+str(n)] = calc_phi(phi_0, nom.m_pflow_jet_phi[i])
            y['jet_MV2c10_h'+str(n)] = nom.m_pflow_jet_flavor_weight_MV2c10[i]
            
            n+=1
        
        n = 0
        for i in topMatches:#bestBtags:#nom.nJets_OR_T):      
            y['top_Pt_'+str(n)] = nom.m_pflow_jet_pt[i]
            y['top_Eta_'+str(n)] = nom.m_pflow_jet_eta[i]
            y['top_E_'+str(n)] = jet4Vecs[i].E()#nom.m_pflow_jet_E[i]
            y['top_Phi_'+str(n)] = calc_phi(phi_0, nom.m_pflow_jet_phi[i])
            y['top_MV2c10_'+str(n)] = nom.m_pflow_jet_flavor_weight_MV2c10[i]
            
            n+=1

        y['MET'] = nom.MET_RefFinal_et
        y['MET_phi'] = calc_phi(phi_0, nom.MET_RefFinal_phi)

        events1l.append(y)

    return decayDicts, events1l, events2l

def make_tensors(inDF, xMax):
    X = inDF #.drop(['higgs_pt'],axis=1)
    X = torch.tensor(X.values, dtype=torch.float32)
    X = X / xMax
    return X #, Y


def pred_pt_1l(X, yMax):
    Y_pred_pt = net_pt1l(X)[:,0]
    return (Y_pred_pt*yMax).float().detach().numpy()

def pred_pt_2l(X, yMax):
    Y_pred_pt = net_pt2l(X)[:,0]
    return (Y_pred_pt*yMax).float().detach().numpy()

def pred_bin_1l(X):
    Y_pred_bin = net_bin1l(X)[:,0]
    return Y_pred_bin.float().detach().numpy()

def pred_bin_2l(X):
    Y_pred_bin = net_bin2l(X)[:,0]
    return Y_pred_bin.float().detach().numpy()


#loop over file list, add prediction branches                                                                                                    
def run_pred(inputPath):
    f = TFile(inputPath, "READ")

    dsid = inputPath.split('/')[-1]
    dsid = dsid.replace('.root', '')
    print(inputPath)

    nom = f.Get('nominal')
    if nom.GetEntries() == 0:
        return 0

    decayDicts, events1l, events2l = create_dict(nom)

    ### Evaluate decay BDT
    decayDF = pd.DataFrame(decayDicts)
    decayMat = xgb.DMatrix(decayDF, feature_names=list(decayDF))
    decayScore = decayModel.predict(decayMat)

    ### Evaluate 1l BDT
    in1l = pd.DataFrame(events1l)
    X_1l = make_tensors(in1l, xMax1l)
    y_pred_pt_1l = pred_pt_1l(X_1l, yMax1l)
    y_pred_bin_1l = pred_bin_1l(X_1l)

    ### Evaluate 2l BDT
    in2l = pd.DataFrame(events2l)
    X_2l = make_tensors(in2l, xMax2l)
    y_pred_pt_2l = pred_pt_2l(X_2l, yMax2l)
    y_pred_bin_2l = pred_bin_2l(X_2l)

    #### Write results to root files

    # Decay mode score
    with root_open(inputPath, mode='a') as myfile:
        decayScore = np.asarray(decayScore)
        decayScore.dtype = [('decayScore', 'float32')]
        decayScore.dtype.names = ['decayScore']
        root_numpy.array2tree(decayScore, tree=myfile.nominal)

        myfile.write()
        myfile.Close()

    # 1l - Semi-leptonic decay
    with root_open(inputPath, mode='a') as myfile:
        dNN_pt_score_3lS = np.asarray(y_pred_pt_1l)
        dNN_pt_score_3lS.dtype = [('dNN_pt_score_3lS', 'float32')]
        dNN_pt_score_3lS.dtype.names = ['dNN_pt_score_3lS']
        root_numpy.array2tree(dNN_pt_score_3lS, tree=myfile.nominal)

        myfile.write()
        myfile.Close()

    with root_open(inputPath, mode='a') as myfile:
        dNN_bin_score_3lS = np.asarray(y_pred_bin_1l)
        dNN_bin_score_3lS.dtype = [('dNN_bin_score_3lS', 'float32')]
        dNN_bin_score_3lS.dtype.names = ['dNN_bin_score_3lS']
        root_numpy.array2tree(dNN_bin_score_3lS, tree=myfile.nominal)
        
        myfile.write()
        myfile.Close()

    # 2l - Fully leptonic decay
    with root_open(inputPath, mode='a') as myfile:
        dNN_pt_score_3lF = np.asarray(y_pred_pt_2l)
        dNN_pt_score_3lF.dtype = [('dNN_pt_score_3lF', 'float32')]
        dNN_pt_score_3lF.dtype.names = ['dNN_pt_score_3lF']
        root_numpy.array2tree(dNN_pt_score_3lF, tree=myfile.nominal)
        
        myfile.write()
        myfile.Close()

    with root_open(inputPath, mode='a') as myfile:
        dNN_bin_score_3lF = np.asarray(y_pred_bin_2l)
        dNN_bin_score_3lF.dtype = [('dNN_bin_score_3lF', 'float32')]
        dNN_bin_score_3lF.dtype.names = ['dNN_bin_score_3lF']
        root_numpy.array2tree(dNN_bin_score_3lF, tree=myfile.nominal)
        
        myfile.write()
        myfile.Close()


linelist = [line.rstrip() for line in open(inf)]
Parallel(n_jobs=20)(delayed(run_pred)(inFile) for inFile in linelist)

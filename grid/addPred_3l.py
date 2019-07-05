import uproot
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
#from rootpy.io import root_open

#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

inf = sys.argv[1]
#outFile = sys.argv[2]
decayModelPath = "models/3l/xgb_decay_decay3l_recoScores.dat"
higgs1lModelPath = "models/3l/xgb_match_higgs1l.dat"
higgs2lModelPath = "models/3l/xgb_match_higgs2l.dat"
topModelPath = "models/3l/xgb_match_top3lJVT.dat"
#outFile = sys.argv[4]

#njet = sys.argv[3]
f = uproot.open(inf)

dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)

oldTree = f.get('nominal')
    #oldTree.SetBranchStatus("*",0)
    #for br in branch_list:
    #    oldTree.SetBranchStatus(br,1)

events1l = []
events2l = []
decayDicts = []
bestScores = []

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

#xgbPtModel = pickle.load(open('models/3l/xgb_higgsPflowFlatBranch.dat

#la=f['nominal'].lazyarrays(['higgs_pt', 'jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*'])#, 'is2LSS0Tau', 'is2LSS1Tau',
la=f['nominal'].lazyarrays(['m_truth*','dilep_type','trilep_type','quadlep_type','higgs*','lep_Pt*','lep_Eta_*', 'lep_E_*','total_charge',
                            'total_leptons', 'lep_Phi*','lep_ID*','lep_Index*', 'm_track_jet*', 'm_pflow_jet*',
                            'nJets_OR_T','nJets_OR_T_MV2c10_70',
                            'MET_RefFinal_et', 'MET_RefFinal_phi'])
#'total_leptons', 'dilep_type', 'total_charge', 'nJets_OR_T_MV2c10_70', 'nJets_OR_T'])

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

def create_dict():
    current = 0
    totalEvt = len(la[b'MET_RefFinal_et'])
    
    for idx in range(len(la[b'MET_RefFinal_et']) ):
        current+=1
        if current%1000==0:
            print(str(current)+"/"+str(totalEvt))
        #if current%100==0:
        #    break

        #if la[b'total_leptons'][idx] < 2: continue
        #if la[b'dilep_type'][idx] < 1: continue                                                                                                
        #if la[b'total_charge'][idx] == 0: continue  
        #if la[b'nJets_OR_T_MV2c10_70'][idx] < 1: continue
        #if la[b'nJets_OR_T'][idx] < 2: continue

        higgCand = LorentzVector()
        
        lep4Vecs = []
        jet4Vecs = []
        
        btags = []
        
        met = LorentzVector()
        met.SetPtEtaPhiE(la[b'MET_RefFinal_et'][idx], 0, la[b'MET_RefFinal_phi'][idx], la[b'MET_RefFinal_et'][idx])

        for i in range(3):
            i = str.encode(str(i))
            lepVec_0 = LorentzVector()
            lepVec_0.SetPtEtaPhiE(la[b'lep_Pt_'+i][idx], la[b'lep_Eta_'+i][idx], la[b'lep_Phi_'+i][idx], 
                                  la[b'lep_E_'+i][idx])
            lep4Vecs.append(lepVec_0)

        for j in range(len(la[b'm_pflow_jet_pt'][idx])):#la[b'selected_jets'][i]:
            jetVec = LorentzVector()
            jetVec.SetPtEtaPhiM(la[b'm_pflow_jet_pt'][idx][j], la[b'm_pflow_jet_eta'][idx][j], la[b'm_pflow_jet_phi'][idx][j], la[b'm_pflow_jet_m'][idx][j])
            jet4Vecs.append(jetVec)
            
            btags.append(la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][j])

        combosTop = []
        
        for l in range(len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]
                
                    t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met, btags[i], btags[j],
                                 la[b'm_pflow_jet_jvt'][idx][i], la[b'm_pflow_jet_jvt'][idx][j],
                                 la[b'm_pflow_jet_numTrk'][idx][i], la[b'm_pflow_jet_numTrk'][idx][j] )
                    
                    combosTop.append([t, comb])

        #loop over combinations, score them in the BDT, figure out the best result
        topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
        topMat = xgb.DMatrix(topDF, feature_names=list(topDF))
        
        topPred = topModel.predict(topMat)
        topBest = np.argmax(topPred)
        
        bestTopComb = combosTop[topBest][1]
        topMatches = bestTopComb[1:]

        combos1l = []

        for l in range(len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]
                    if l==0:
                        k = higgs1lDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, 
                                         la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][i], la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][j], lep4Vecs[1],
                                         la[b'm_pflow_jet_jvt'][idx][i], la[b'm_pflow_jet_jvt'][idx][j],
                                         la[b'm_pflow_jet_numTrk'][idx][i], la[b'm_pflow_jet_numTrk'][idx][j])
                    else:
                        k = higgs1lDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, 
                                         la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][i], la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][j], lep4Vecs[0],
                                         la[b'm_pflow_jet_jvt'][idx][i], la[b'm_pflow_jet_jvt'][idx][j],
                                         la[b'm_pflow_jet_numTrk'][idx][i], la[b'm_pflow_jet_numTrk'][idx][j])
                        
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
        k['nJets'] = la[b'nJets_OR_T'][idx]
        k['nJets_MV2c10_70'] = la[b'nJets_OR_T_MV2c10_70'][idx]
        k['higgs2l_score'] = pred2l[best2l]
        k['higgs1l_score'] = pred1l[best1l]
        decayDicts.append(k)


        ### Add 2l pt prediction dict

        q = {}
        q['comboScore'] = pred2l[best2l]
        
        if lepMatch2l == 1:
            q['lep_Pt_0'] = la[b'lep_Pt_0'][idx]
            q['lep_Eta_0'] = la[b'lep_Eta_0'][idx]
            phi_0 = la[b'lep_Phi_0'][idx]
            q['lep_E_0'] = la[b'lep_E_0'][idx]

            q['lep_Pt_1'] = la[b'lep_Pt_1'][idx]
            q['lep_Eta_1'] = la[b'lep_Eta_1'][idx]
            q['lep_Phi_1'] = calc_phi(phi_0, la[b'lep_Phi_1'][idx])
            q['lep_E_1'] = la[b'lep_E_1'][idx]
            
            q['lep_Pt_2'] = la[b'lep_Pt_2'][idx]
            q['lep_Eta_2'] = la[b'lep_Eta_2'][idx]
            q['lep_Phi_2'] = calc_phi(phi_0, la[b'lep_Phi_2'][idx])
            q['lep_E_2'] = la[b'lep_E_2'][idx]
            
        elif lepMatch2l == 2:
            q['lep_Pt_0'] = la[b'lep_Pt_0'][idx]
            q['lep_Eta_0'] = la[b'lep_Eta_0'][idx]
            phi_0 = la[b'lep_Phi_0'][idx]
            q['lep_E_0'] = la[b'lep_E_0'][idx]
            
            q['lep_Pt_1'] = la[b'lep_Pt_2'][idx]
            q['lep_Eta_1'] = la[b'lep_Eta_2'][idx]
            q['lep_Phi_1'] = calc_phi(phi_0, la[b'lep_Phi_2'][idx])
            q['lep_E_1'] = la[b'lep_E_2'][idx]
            
            q['lep_Pt_2'] = la[b'lep_Pt_1'][idx]
            q['lep_Eta_2'] = la[b'lep_Eta_1'][idx]
            q['lep_Phi_2'] = calc_phi(phi_0, la[b'lep_Phi_1'][idx])
            q['lep_E_2'] = la[b'lep_E_1'][idx]

        n = 0
        for i in topMatches:
            q['top_Pt_'+str(n)] = la[b'm_pflow_jet_pt'][idx][i]
            q['top_Eta_'+str(n)] = la[b'm_pflow_jet_eta'][idx][i]
            q['top_E_'+str(n)] = jet4Vecs[i].E()#la[b'm_pflow_jet_E'][idx][i]
            q['top_Phi_'+str(n)] = calc_phi(phi_0, la[b'm_pflow_jet_phi'][idx][i])
            q['top_MV2c10_'+str(n)] = la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][i]

            n+=1

        q['MET'] = la[b'MET_RefFinal_et'][idx]
        q['MET_phi'] = calc_phi(phi_0, la[b'MET_RefFinal_phi'][idx])

        events2l.append(q)

        ### Add 1l Pt prediction dict

        y = {}
        #y['higgs_pt'] = la[b'higgs_pt'][idx]
        y['comboScore'] = pred1l[best1l]
        
        if lepMatch1l == 1:
            y['lep_Pt_H'] = la[b'lep_Pt_1'][idx]
            y['lep_Eta_H'] = la[b'lep_Eta_1'][idx]
            phi_0 = la[b'lep_Phi_1'][idx]
            y['lep_E_H'] = la[b'lep_E_1'][idx]
            
            y['lep_Pt_0'] = la[b'lep_Pt_0'][idx]
            y['lep_Eta_0'] = la[b'lep_Eta_0'][idx]
            y['lep_Phi_0'] = calc_phi(phi_0, la[b'lep_Phi_0'][idx])
            y['lep_E_0'] = la[b'lep_E_0'][idx]
            
            y['lep_Pt_1'] = la[b'lep_Pt_2'][idx]
            y['lep_Eta_1'] = la[b'lep_Eta_2'][idx]
            y['lep_Phi_1'] = calc_phi(phi_0, la[b'lep_Phi_2'][idx])
            y['lep_E_1'] = la[b'lep_E_2'][idx]

        elif lepMatch1l == 2:
            y['lep_Pt_H'] = la[b'lep_Pt_2'][idx]
            y['lep_Eta_H'] = la[b'lep_Eta_2'][idx]
            phi_0 = la[b'lep_Phi_2'][idx]
            y['lep_E_H'] = la[b'lep_E_2'][idx]
            
            y['lep_Pt_0'] = la[b'lep_Pt_0'][idx]
            y['lep_Eta_0'] = la[b'lep_Eta_0'][idx]
            y['lep_Phi_0'] = calc_phi(phi_0, la[b'lep_Phi_0'][idx])
            y['lep_E_0'] = la[b'lep_E_0'][idx]
            
            y['lep_Pt_1'] = la[b'lep_Pt_1'][idx]
            y['lep_Eta_1'] = la[b'lep_Eta_1'][idx]
            y['lep_Phi_1'] = calc_phi(phi_0, la[b'lep_Phi_1'][idx])
            y['lep_E_1'] = la[b'lep_E_1'][idx]
            
        n = 0
        for i in jetMatches1l:#la[b'nJets_OR_T):
            
            y['jet_Pt_h'+str(n)] = la[b'm_pflow_jet_pt'][idx][i]
            y['jet_Eta_h'+str(n)] = la[b'm_pflow_jet_eta'][idx][i]
            y['jet_E_h'+str(n)] = jet4Vecs[i].E()#la[b'm_pflow_jet_E'][idx][i]
            y['jet_Phi_h'+str(n)] = calc_phi(phi_0, la[b'm_pflow_jet_phi'][idx][i])
            y['jet_MV2c10_h'+str(n)] = la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][i]
            
            n+=1
        
        n = 0
        for i in topMatches:#bestBtags:#la[b'nJets_OR_T):      
            y['top_Pt_'+str(n)] = la[b'm_pflow_jet_pt'][idx][i]
            y['top_Eta_'+str(n)] = la[b'm_pflow_jet_eta'][idx][i]
            y['top_E_'+str(n)] = jet4Vecs[i].E()#la[b'm_pflow_jet_E'][idx][i]
            y['top_Phi_'+str(n)] = calc_phi(phi_0, la[b'm_pflow_jet_phi'][idx][i])
            y['top_MV2c10_'+str(n)] = la[b'm_pflow_jet_flavor_weight_MV2c10'][idx][i]
            
            n+=1

        y['MET'] = la[b'MET_RefFinal_et'][idx]
        y['MET_phi'] = calc_phi(phi_0, la[b'MET_RefFinal_phi'][idx])

        events1l.append(y)

    return decayDicts, events1l, events2l

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

    X = inDF #.drop(['higgs_pt'],axis=1)
    
    X = torch.tensor(X.values, dtype=torch.float32)
    
    X = X / xMax
    
    return X #, Y


def pred_pt_1l(X, yMax):

    print(X.shape[1])
    net_pt = Net(X.size()[1],100,6)
    net_pt.load_state_dict(torch.load('models/3l/model_higgs1l_6l_100n.pt'))
    net_pt.eval()

    Y_pred_pt = net_pt(X)[:,0]

    return (Y_pred_pt*yMax).float().detach().numpy()

def pred_pt_2l(X, yMax):

    print(X.shape[1])
    net_pt = Net(X.size()[1],100,5)
    net_pt.load_state_dict(torch.load('models/3l/model_higgs2l_5l_100n.pt'))
    net_pt.eval()

    Y_pred_pt = net_pt(X)[:,0]

    return (Y_pred_pt*yMax).float().detach().numpy()

def pred_bin_1l(X):

    net_bin = Net(X.size()[1], 75, 6)
    net_bin.load_state_dict(torch.load('models/3l/model_higgs1lBin_6l_75n.pt'))
    net_bin.eval()
    
    Y_pred_bin = net_bin(X)[:,0]

    return Y_pred_bin.float().detach().numpy()

def pred_bin_2l(X):

    net_bin = Net(X.size()[1], 100, 6)
    net_bin.load_state_dict(torch.load('models/3l/model_higgs2lBin_6l_100n.pt'))
    net_bin.eval()

    Y_pred_bin = net_bin(X)[:,0]

    return Y_pred_bin.float().detach().numpy()


decayDicts, events1l, events2l = create_dict()

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
with root_open(inf, mode='a') as myfile:
    decayScore = np.asarray(decayScore)
    decayScore.dtype = [('decayScore', 'float32')]
    decayScore.dtype.names = ['decayScore']
    root_numpy.array2tree(decayScore, tree=myfile.nominal)

    myfile.write()
    myfile.Close()


# 1l - Semi-leptonic decay
with root_open(inf, mode='a') as myfile:
    dNN_pt_score_3lS = np.asarray(y_pred_pt_1l)
    dNN_pt_score_3lS.dtype = [('dNN_pt_score_3lS', 'float32')]
    dNN_pt_score_3lS.dtype.names = ['dNN_pt_score_3lS']
    root_numpy.array2tree(dNN_pt_score_3lS, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

with root_open(inf, mode='a') as myfile:
    dNN_bin_score_3lS = np.asarray(y_pred_bin_1l)
    dNN_bin_score_3lS.dtype = [('dNN_bin_score_3lS', 'float32')]
    dNN_bin_score_3lS.dtype.names = ['dNN_bin_score_3lS']
    root_numpy.array2tree(dNN_bin_score_3lS, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

# 2l - Fully leptonic decay
with root_open(inf, mode='a') as myfile:
    dNN_pt_score_3lF = np.asarray(y_pred_pt_2l)
    dNN_pt_score_3lF.dtype = [('dNN_pt_score_3lF', 'float32')]
    dNN_pt_score_3lF.dtype.names = ['dNN_pt_score_3lF']
    root_numpy.array2tree(dNN_pt_score_3lF, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

with root_open(inf, mode='a') as myfile:
    dNN_bin_score_3lF = np.asarray(y_pred_bin_2l)
    dNN_bin_score_3lF.dtype = [('dNN_bin_score_3lF', 'float32')]
    dNN_bin_score_3lF.dtype.names = ['dNN_bin_score_3lF']
    root_numpy.array2tree(dNN_bin_score_3lF, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

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
higgsModelPath = "models/xgb_match_higgsPflowFlat.dat"
topModelPath = "models/xgb_match_topPflow.dat"
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

events = []
bestScores = []

xgbModel = pickle.load(open(higgsModelPath, "rb"))

topModel = pickle.load(open(topModelPath, "rb"))

normFactors = np.load('models/normFactors.npy')
normFactors = torch.from_numpy(normFactors).float()
yMax = normFactors[0]#torch.from_numpy(normFactors[0]).float()#Y.max(0, keepdim=True)[0]
xMax = normFactors[1:]#torch.from_numpy(normFactors[1:]).float()#X.max(0, keepdim=True)[0]

#xgbPtModel = pickle.load(open('models/xgb_higgsPflowFlatBranch.dat

#la=f['nominal'].lazyarrays(['higgs_pt', 'jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*'])#, 'is2LSS0Tau', 'is2LSS1Tau',
la=f['nominal'].lazyarrays(['m_truth*','dilep_type','trilep_type','quadlep_type','higgs*','lep_Pt*','lep_Eta_*', 'lep_E_*','total_charge',
                            'total_leptons', 'lep_Phi*','lep_ID*','lep_Index*', 'm_track_jet*', 'm_jet*',
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

        #for i in range(2):
        lep_Pt_0 = la[b'lep_Pt_0'][idx]
        lep_Eta_0 = la[b'lep_Eta_0'][idx]
        lep_Phi_0 = la[b'lep_Phi_0'][idx]
        lep_E_0 = la[b'lep_E_0'][idx]
        
        lepVec_0 = LorentzVector()
        lepVec_0.SetPtEtaPhiE(lep_Pt_0, lep_Eta_0, lep_Phi_0, lep_E_0)
        lep4Vecs.append(lepVec_0)

        lep_Pt_1 = la[b'lep_Pt_1'][idx]
        lep_Eta_1 = la[b'lep_Eta_1'][idx]
        lep_Phi_1 = la[b'lep_Phi_1'][idx]
        lep_E_1 = la[b'lep_E_1'][idx]

        lepVec_1 = LorentzVector()
        lepVec_1.SetPtEtaPhiE(lep_Pt_1, lep_Eta_1, lep_Phi_1, lep_E_1)
        lep4Vecs.append(lepVec_1)

        for j in range(len(la[b'm_jet_pt'][idx])):#la[b'selected_jets'][i]:
            jetVec = LorentzVector()
            jetVec.SetPtEtaPhiE(la[b'm_jet_pt'][idx][j], la[b'm_jet_eta'][idx][j], la[b'm_jet_phi'][idx][j], la[b'm_jet_E'][idx][j])
            jet4Vecs.append(jetVec)
            
            btags.append(la[b'm_jet_flavor_weight_MV2c10'][idx][j])

        combos = []
        combosTop = []
        
        for l in range(len(lep4Vecs)):
            for i in range(len(jet4Vecs)-1):
                for j in range(i+1, len(jet4Vecs)):
                    comb = [l,i,j]
                
                    t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], met, btags[i], btags[j] )

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
                        k = higgsDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, btags[i], btags[j], lep4Vecs[1] )
                    else:
                        k = higgsDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, btags[i], btags[j], lep4Vecs[0] )

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
        k['higgs_pt'] = la[b'higgs_pt'][idx]
        k['comboScore'] = pred[best]
        
        if lepMatch == 0:
            k['lep_Pt_H'] = la[b'lep_Pt_0'][idx]
            k['lep_Eta_H'] = la[b'lep_Eta_0'][idx]
            phi_0 = la[b'lep_Phi_0'][idx]
            k['lep_E_H'] = la[b'lep_E_0'][idx]
            
            k['lep_Pt_O'] = la[b'lep_Pt_1'][idx]
            k['lep_Eta_O'] = la[b'lep_Eta_1'][idx]
            k['lep_Phi_O'] = calc_phi(phi_0, la[b'lep_Phi_1'][idx])
            k['lep_E_O'] = la[b'lep_E_1'][idx]

        elif lepMatch == 1:
            k['lep_Pt_H'] = la[b'lep_Pt_1'][idx]
            k['lep_Eta_H'] = la[b'lep_Eta_1'][idx]
            phi_0 = la[b'lep_Phi_1'][idx]
            k['lep_E_H'] = la[b'lep_E_1'][idx]

            k['lep_Pt_O'] = la[b'lep_Pt_0'][idx]
            k['lep_Eta_O'] = la[b'lep_Eta_0'][idx]
            k['lep_Phi_O'] = calc_phi(phi_0, la[b'lep_Phi_0'][idx])
            k['lep_E_O'] = la[b'lep_E_0'][idx]

            
        n = 0
        for i in jetMatches:#la[b'nJets_OR_T):
            
            k['jet_Pt_h'+str(n)] = la[b'm_jet_pt'][idx][i]
            k['jet_Eta_h'+str(n)] = la[b'm_jet_eta'][idx][i]
            k['jet_E_h'+str(n)] = la[b'm_jet_E'][idx][i]
            k['jet_Phi_h'+str(n)] = calc_phi(phi_0, la[b'm_jet_phi'][idx][i])
            k['jet_MV2c10_h'+str(n)] = la[b'm_jet_flavor_weight_MV2c10'][idx][i]
            
            n+=1

        btags = np.array(btags)
            
        btags[jetMatches[0]] = 0
        btags[jetMatches[1]] = 0
        bestBtags = np.argpartition(btags, -2)[-2:]
        
        n = 0
        for i in topMatches:#bestBtags:#la[b'nJets_OR_T):      
            k['top_Pt_'+str(n)] = la[b'm_jet_pt'][idx][i]
            k['top_Eta_'+str(n)] = la[b'm_jet_eta'][idx][i]
            k['top_E_'+str(n)] = la[b'm_jet_E'][idx][i]
            k['top_Phi_'+str(n)] = calc_phi(phi_0, la[b'm_jet_phi'][idx][i])
            k['top_MV2c10_'+str(n)] = la[b'm_jet_flavor_weight_MV2c10'][idx][i]
            
            n+=1

        k['MET'] = la[b'MET_RefFinal_et'][idx]
        k['MET_phi'] = calc_phi(phi_0, la[b'MET_RefFinal_phi'][idx])

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


def make_tensors(inDF, yMax, xMax):
    
    #inDF['higgs_pt'] = pd.cut(inDF['higgs_pt'], bins=[0, 150000, 9999999999], labels=[0,1])
    
    Y = inDF['higgs_pt']
    X = inDF.drop(['higgs_pt'],axis=1)
    
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = torch.FloatTensor(Y.values)
    
    #yMax = torch.from_numpy(normFactors[0]).float()#Y.max(0, keepdim=True)[0]
    #xMax = torch.from_numpy(normFactors[1:]).float()#X.max(0, keepdim=True)[0]
    
    X = X / xMax
    Y = Y / yMax
    
    return X, Y


def pred_pt(X, Y, yMax):

    print(X.shape[1])
    net_pt = Net(X.size()[1],75,6)
    net_pt.load_state_dict(torch.load('models/torch_pt_6l_75n.pt'))
    net_pt.eval()

    Y_pred_pt = net_pt(X)[:,0]

    return (Y_pred_pt*yMax).float().detach().numpy()


def pred_bin(X, Y):

    net_bin = Net(X.size()[1], 75, 5)
    net_bin.load_state_dict(torch.load('models/torch_binned_5l_75n.pt'))
    net_bin.eval()
    
    Y_pred_bin = net_bin(X)[:,0]

    return Y_pred_bin.float().detach().numpy()


event_dict = create_dict()

inDF = pd.DataFrame(event_dict)

X, Y = make_tensors(inDF, yMax, xMax)

y_pred_pt = pred_pt(X, Y, yMax)
y_pred_bin = pred_bin(X, Y)

inDF['y_pred_pt'] = y_pred_pt
inDF['y_pred_bin'] = y_pred_bin

inDF.to_csv('test.csv', index=False)

print(len(y_pred_pt), len(y_pred_bin))
#f.Close()

#y_pred_pt = y_pred_pt.float().detach().numpy()
#y_pred_bin = y_pred_bin.float().detach().numpy()

with root_open(inf, mode='a') as myfile:
    dNN_pt_score = np.asarray(y_pred_pt)
    dNN_pt_score.dtype = [('dNN_pt_score', 'float32')]
    dNN_pt_score.dtype.names = ['dNN_pt_score']
    root_numpy.array2tree(dNN_pt_score, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

with root_open(inf, mode='a') as myfile:
    dNN_bin_score = np.asarray(y_pred_bin)
    dNN_bin_score.dtype = [('dNN_bin_score', 'float32')]
    dNN_bin_score.dtype.names = ['dNN_bin_score']
    root_numpy.array2tree(dNN_bin_score, tree=myfile.nominal)

    myfile.write()
    myfile.Close()

import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math

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

xgbModelPath = "models/sigBkg/3l_inc.dat"
xgbModel = pickle.load(open(xgbModelPath, "rb"))

events = []

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
for e in oldTree:
    current+=1
    if current%10000==0:
        print(current)

    #if e.total_leptons!=2: continue
    #if abs(e.total_charge)!=2: continue
    #if e.nJets_OR_T_MV2c10_70<1: continue
    #if e.nJets_OR_T<4: continue

    if e.is3L==0: continue
    if e.trilep_type==0: continue
    if abs(e.total_charge)!=1: continue
    if e.nJets_OR_T<2: continue
    if e.nJets_OR_T_MV2c10_70<1: continue
    if e.lep_ID_0==-e.lep_ID_1 and abs(e.Mll01-91.2e3)<10e3: continue
    if e.lep_ID_0==-e.lep_ID_2 and abs(e.Mll02-91.2e3)<10e3: continue

    k = {}

    k['trilep_type'] = e.dilep_type
    k['lep_Pt_0'] = e.lep_Pt_0
    k['lep_Eta_0'] = e.lep_Eta_0
    k['lep_Phi_0'] = e.lep_Phi_0
    k['lep_ID_0'] = e.lep_ID_0
    k['lep_Pt_1'] = e.lep_Pt_1
    k['lep_Eta_1'] = e.lep_Eta_1
    k['lep_Phi_1'] = e.lep_Phi_1
    k['lep_ID_1'] = e.lep_ID_1
    k['lep_Pt_2'] = e.lep_Pt_2
    k['lep_Eta_2'] = e.lep_Eta_2
    k['lep_Phi_2'] = e.lep_Phi_2
    k['lep_ID_2'] = e.lep_ID_2
    k['Mll01'] = e.Mll01
    k['Mll02'] = e.Mll02
    k['Mll12'] = e.Mll12
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

    k['dNN_bin_score_3l'] = e.dNN_bin_score_3l
    k['dNN_pt_score_3l'] = e.dNN_pt_score_3l
    k['decayScore'] = e.decayScore

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)                                                                       
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)

    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
xgbMat = xgb.DMatrix(dFrame, feature_names=list(dFrame))
xgbScore3l = xgbModel.predict(xgbMat)

# Decay mode score                                                                                                                              
with root_open(inf, mode='a') as myfile:
    xgbScore3l = np.asarray(xgbScore3l)
    xgbScore3l.dtype = [('xgbScore3l', 'float32')]
    xgbScore3l.dtype.names = ['xgbScore3l']
    root_numpy.array2tree(xgbScore3l, tree=myfile.nominal)

    myfile.write()
    myfile.Close()


'''
Creates dict with features for distinguishing signal and background events
'''


import math

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

def sigBkgDict2l(nom, signal=-1):
    k = {}

    if signal!=-1:
        k['signal'] = signal
        k['weight'] = nom.weight

    k['dilep_type'] = nom.dilep_type
    k['lep_Pt_0'] = nom.lep_Pt_0
    k['lep_Eta_0'] = nom.lep_Eta_0
    phi_0 = nom.lep_Phi_0
    #k['lep_ID_0'] = nom.lep_ID_0
    k['lep_Pt_1'] = nom.lep_Pt_1
    k['lep_Eta_1'] = nom.lep_Eta_1
    k['lep_Phi_1'] = calc_phi(phi_0, nom.lep_Phi_1)
    #k['lep_ID_1'] = nom.lep_ID_1
    k['Mll01'] = nom.Mll01
    k['DRll01'] = nom.DRll01
    #k['Ptll01'] = nom.Ptll01
    k['lead_jetPt'] = nom.lead_jetPt
    k['lead_jetEta'] = nom.lead_jetEta
    k['lead_jetPhi'] = calc_phi(phi_0, nom.lead_jetPhi)
    k['sublead_jetPt'] = nom.sublead_jetPt
    k['sublead_jetEta'] = nom.sublead_jetEta
    k['sublead_jetPhi'] = calc_phi(phi_0, nom.sublead_jetPhi)
    k['HT'] = nom.HT

    k['nJets_OR_T'] = nom.nJets_OR_T
    k['nJets_OR_T_MV2c10_70'] = nom.nJets_OR_T_MV2c10_70
    k['MET_RefFinal_et'] = nom.MET_RefFinal_et
    k['MET_RefFinal_phi'] = calc_phi(phi_0, nom.MET_RefFinal_phi)

    k['DRlj00'] = nom.DRlj00
    k['DRjj01'] = nom.DRjj01
    k['min_DRl0j'] = nom.min_DRl0j
    k['min_DRl1j'] = nom.min_DRl1j

    k['pt_score'] = nom.recoHiggsPt_2lSS
    k['higgsScore'] = nom.higgsRecoScore2lSS
    k['topScore'] = nom.topRecoScore

    return k

def sigBkgDict3l(nom, signal=-1):
    k = {}

    if signal!=-1:
        k['signal'] = signal
        k['scale_nom'] = nom.weight

    k['trilep_type'] = nom.trilep_type

    k['lep_Pt_0'] = nom.lep_Pt_0
    k['lep_Eta_0'] = nom.lep_Eta_0
    phi_0 = nom.lep_Phi_0

    k['lep_Pt_1'] = nom.lep_Pt_1
    k['lep_Eta_1'] = nom.lep_Eta_1
    k['lep_Phi_1'] = calc_phi(phi_0, nom.lep_Phi_1)

    k['lep_Pt_2'] = nom.lep_Pt_2
    k['lep_Eta_2'] = nom.lep_Eta_2
    k['lep_Phi_2'] = calc_phi(phi_0, nom.lep_Phi_2)

    k['Mll01'] = nom.Mll01
    k['Mll02'] = nom.Mll02
    k['Mll12'] = nom.Mll12

    k['DRll01'] = nom.DRll01
    k['DRll02'] = nom.DRll02
    k['DRll12'] = nom.DRll12

    k['lead_jetPt'] = nom.jet_pt[0]
    k['lead_jetEta'] = nom.jet_eta[0]
    k['lead_jetPhi'] = calc_phi(phi_0, nom.jet_phi[0])

    k['sublead_jetPt'] = nom.jet_pt[1]
    k['sublead_jetEta'] = nom.jet_eta[1]
    k['sublead_jetPhi'] = calc_phi(phi_0, nom.jet_phi[1])

    k['HT'] = nom.HT
    k['nJets_OR'] = nom.nJets_OR_fixed
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70_fixed
    k['met'] = nom.met_met
    k['met_phi'] = calc_phi(phi_0, nom.met_phi)
    k['MLepMet'] = nom.MLepMet

    k['minDeltaR_LJ_0'] = nom.minDeltaR_LJ_0
    k['minDeltaR_LJ_1'] = nom.minDeltaR_LJ_1
    k['minDeltaR_LJ_0'] = nom.minDeltaR_LJ_2

    k['DeltaR_min_lep_jet'] = nom.DeltaR_min_lep_jet
    k['DeltaR_min_lep_bjet'] = nom.DeltaR_min_lep_bjet
    k['mjjMax_frwdJet'] = nom.mjjMax_frwdJet

    k['Mlll012'] = nom.Mlll012

    k['pt_score_3lF'] = nom.recoHiggsPt_3lF
    k['pt_score_3lS'] = nom.recoHiggsPt_3lS
    k['decayScore'] = nom.decayScore

    k['higgsRecoScore1l'] = nom.higgsRecoScore_3lF
    k['higgsScore2l'] = nom.higgsRecoScore_3lS
    k['topScore'] = nom.topRecoScore

    return k

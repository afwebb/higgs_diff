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
    k['lep_Phi_0'] = calc_phi(nom.met_phi, nom.lep_Phi_0)

    k['lep_Pt_1'] = nom.lep_Pt_1
    k['lep_Eta_1'] = nom.lep_Eta_1
    k['lep_Phi_1'] = calc_phi(nom.met_phi, nom.lep_Phi_1)

    k['Ml0l1'] = nom.Mll01
    k['dR_l0_l1'] = nom.DRll01

    k['jet_Pt_0'] = nom.jet_pt[0]
    k['jet_Eta_0'] = nom.jet_eta[0]
    k['jet_Phi_0'] = calc_phi(nom.met_phi, nom.jet_phi[0])

    k['jet_Pt_1'] = nom.jet_pt[1]
    k['jet_Eta_1'] = nom.jet_eta[1]
    k['jet_Phi_1'] = calc_phi(nom.met_phi, nom.jet_phi[1])

    k['HT'] = nom.HT
    k['nJets_OR'] = nom.nJets_OR_fixed
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70_fixed
    k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85
    k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    k['met'] = nom.met_met
    k['MLepMet'] = nom.MLepMet

    k['min_dR_l0_jet'] = nom.minDeltaR_LJ_0
    k['min_dR_l1_jet'] = nom.minDeltaR_LJ_1

    #k['min_dR_lep_jet'] = nom.DeltaR_min_lep_jet
    k['min_dR_lep_bjet'] = nom.DeltaR_min_lep_bjet
    k['mjjMax_frwdJet'] = nom.mjjMax_frwdJet

    k['binHiggsPt_2lSS'] = nom.binHiggsPt_2lSS
    k['higgsRecoScore'] = nom.higgsRecoScore2lSS
    k['topRecoScore'] = nom.topRecoScore

    return k

def sigBkgDict3l(nom, signal=-1):
    k = {}

    if signal!=-1:
        k['signal'] = signal
        k['weight'] = nom.weight

    k['trilep_type'] = nom.trilep_type

    k['lep_Pt_0'] = nom.lep_Pt_0
    k['lep_Eta_0'] = nom.lep_Eta_0
    k['lep_Phi_0'] = calc_phi(nom.met_phi, nom.lep_Phi_0)

    k['lep_Pt_1'] = nom.lep_Pt_1
    k['lep_Eta_1'] = nom.lep_Eta_1
    k['lep_Phi_1'] = calc_phi(nom.met_phi, nom.lep_Phi_1)

    k['lep_Pt_2'] = nom.lep_Pt_2
    k['lep_Eta_2'] = nom.lep_Eta_2
    k['lep_Phi_2'] = calc_phi(nom.met_phi, nom.lep_Phi_2)

    k['Ml0l1'] = nom.Mll01
    k['Ml0l2'] = nom.Mll02
    k['Ml1l2'] = nom.Mll12

    k['dR_l0_l1'] = nom.DRll01
    k['dR_l0_l2'] = nom.DRll02
    k['dR_l1_l2'] = nom.DRll12

    k['jet_Pt_0'] = nom.jet_pt[0]
    k['jet_Eta_0'] = nom.jet_eta[0]
    k['jet_Phi_0'] = calc_phi(nom.met_phi, nom.jet_phi[0])

    k['jet_Pt_1'] = nom.jet_pt[1]
    k['jet_Eta_1'] = nom.jet_eta[1]
    k['jet_Phi_1'] = calc_phi(nom.met_phi, nom.jet_phi[1])

    #k['HT'] = nom.HT
    k['nJets_OR'] = nom.nJets_OR_fixed
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70_fixed
    k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85
    k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    k['met'] = nom.met_met
    #k['met_phi'] = calc_phi(phi_0, nom.met_phi)
    k['MLepMet'] = nom.MLepMet

    k['min_dR_l0_jet'] = nom.minDeltaR_LJ_0
    k['min_dR_l1_jet'] = nom.minDeltaR_LJ_1
    k['min_dR_l2_jet'] = nom.minDeltaR_LJ_2

    #k['DeltaR_min_lep_jet'] = nom.DeltaR_min_lep_jet
    k['min_dR_lep_bjet'] = nom.DeltaR_min_lep_bjet
    k['mjjMax_frwdJet'] = nom.mjjMax_frwdJet

    k['Ml0l1l2'] = nom.Mlll012

    k['binHiggsPt_3lF'] = nom.binHiggsPt_3lF
    k['binHiggsPt_3lS'] = nom.binHiggsPt_3lS
    k['decayScore'] = nom.decayScore

    k['higgsRecoScore3lF'] = nom.higgsRecoScore3lF
    k['higgsRecoScore3lS'] = nom.higgsRecoScore3lS
    k['topScore'] = nom.topRecoScore

    return k

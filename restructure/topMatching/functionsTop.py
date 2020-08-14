import ROOT
from rootpy.vector import LorentzVector
from dictTop import topDictFlat2lSS, topDictFourVec2lSS, topDictFlat3l, topDictFourVec3l

def selection2lSS(nom):
    ''' Takes a root tree event. Returns true if event passes 2lSS selection, returns false otherwise '''

    if nom.total_leptons!=2: return False
    elif nom.total_charge==0: return False
    elif nom.dilep_type<1: return False
    elif nom.nJets_OR<4: return False
    elif nom.nJets_OR_DL1r_70<1: return False
    elif nom.lep_Pt_0<20000: return False
    elif nom.lep_Pt_1<20000: return False
    else: return True

def selection3l(nom):
    ''' Takes a root tree event. Returns true if event passes 3l selection, returns false otherwise '''

    if nom.trilep_type==0: return False
    elif nom.nJets_OR<2: return False
    elif nom.nJets_OR_DL1r_70==0: return False
    elif nom.total_leptons!=3: return False
    elif nom.lep_Pt_0<10000: return False
    elif nom.lep_Pt_1<15000: return False
    elif nom.lep_Pt_2<15000: return False
    else: return True

def convertDL1r(nom, i):
    ''' Takes a ttree and an index, return an integer for the DL1r WP for the jet at index i '''
    # Probably obselete - branch jet_tagWeightBin_DL1r_Continuous seems to do this already

    if nom.jet_isbtagged_DL1r_60[i]=='\x01':
        return 4
    elif nom.jet_isbtagged_DL1r_70[i]=='\x01':
        return 3
    elif nom.jet_isbtagged_DL1r_77[i]=='\x01':
        return 2
    elif nom.jet_isbtagged_DL1r_85[i]=='\x01':
        return 1
    else:
        return 0

def jetCombos2lSS(nom, withMatch):
    ''' 
    Takes a ttree, returns an array of tuples of (dict, jet indices) for top2lSS matching for all possible combinations of jets 
    varType should be either flat or fourVec
    '''

    #Define met, lepton Lorentz Vectors
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met_met, 0, nom.met_phi, nom.met_met)

    lep0 = LorentzVector()
    lep0.SetPtEtaPhiE(nom.lep_Pt_0, nom.lep_Eta_0, nom.lep_Phi_0, nom.lep_E_0)

    lep1 = LorentzVector()
    lep1.SetPtEtaPhiE(nom.lep_Pt_1, nom.lep_Eta_1, nom.lep_Phi_1, nom.lep_E_1)

    #Loop over jet pairs, add topDict to combosTop for each one
    combosTop = {'flatDicts':[],'fourVecDicts':[],'jetIdx':[],'truthComb':[]}
    #truthComb = []
    for i in range(len(nom.jet_pt)-1):
        if nom.jet_jvt[i]<0.59: continue #Only include jets that pass JVT cut
        for j in range(i+1, len(nom.jet_pt)):
            combosTop['jetIdx'].append([i,j])

            #Check if this combination of jets are truth Bs
            isTop = 0
            if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthPartonLabel[i])==5:
                if abs(nom.jet_parents[j])==6 and abs(nom.jet_truthPartonLabel[j])==5:
                    isTop = 1
                    combosTop['truthComb'] = [i,j]
            

            jet0 = LorentzVector()
            jet0.SetPtEtaPhiE(nom.jet_pt[i], nom.jet_eta[i], nom.jet_phi[i], nom.jet_e[i])
            jet1 = LorentzVector()
            jet1.SetPtEtaPhiE(nom.jet_pt[j], nom.jet_eta[j], nom.jet_phi[j], nom.jet_e[j])

            if withMatch:
                t = topDictFlat2lSS( jet0, jet1, lep0, lep1, met, nom.jet_tagWeightBin_DL1r_Continuous[i], 
                                     nom.jet_tagWeightBin_DL1r_Continuous[j], isTop )
                combosTop['flatDicts'].append(t)
                t = topDictFourVec2lSS( jet0, jet1, lep0, lep1, met, nom.jet_tagWeightBin_DL1r_Continuous[i], 
                                        nom.jet_tagWeightBin_DL1r_Continuous[j], isTop )
                combosTop['fourVecDicts'].append(t)
            else:
                t = topDictFlat2lSS( jet0, jet1, lep0, lep1, met, nom.jet_tagWeightBin_DL1r_Continuous[i], 
                                     nom.jet_tagWeightBin_DL1r_Continuous[j] )     
                combosTop['flatDicts'].append(t)                                                                   
                t = topDictFourVec2lSS( jet0, jet1, lep0, lep1, met, nom.jet_tagWeightBin_DL1r_Continuous[i], 
                                        nom.jet_tagWeightBin_DL1r_Continuous[j] )
                combosTop['fourVecDicts'].append(t)
            
    return combosTop #, truthComb

def jetCombos3l(nom, withMatch):
    '''                                                                                                                          Takes a ttree, returns an array of tuples of (dict, jet indices) for top2lSS matching for all possible combinations of 
    jets. withMatch option determines whether to include the "match" 
    '''

    #Define met, lepton Lorentz Vectors                                                                     
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met_met, 0, nom.met_phi, nom.met_met)

    lep0 = LorentzVector()
    lep0.SetPtEtaPhiE(nom.lep_Pt_0, nom.lep_Eta_0, nom.lep_Phi_0, nom.lep_E_0)

    lep1 = LorentzVector()
    lep1.SetPtEtaPhiE(nom.lep_Pt_1, nom.lep_Eta_1, nom.lep_Phi_1, nom.lep_E_1)

    lep2 = LorentzVector()
    lep2.SetPtEtaPhiE(nom.lep_Pt_2, nom.lep_Eta_2, nom.lep_Phi_2, nom.lep_E_2)

    #Loop over jet pairs, add topDict to combosTop for each one                                                      
    combosTop = {'flatDicts':[],'fourVecDicts':[],'jetIdx':[],'truthComb':[]}                                     
    for i in range(len(nom.jet_pt)-1):
        for j in range(i+1, len(nom.jet_pt)):                                                                          
            combosTop['jetIdx'].append([i,j])
            isTop = 0                                                                                                
            if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthPartonLabel[i])==5:                                  
                if abs(nom.jet_parents[j])==6 and abs(nom.jet_truthPartonLabel[j])==5:
                    isTop = 1
                    combosTop['truthComb'] = [i,j]
                    
            jet0 = LorentzVector()                                                                                 
            jet0.SetPtEtaPhiE(nom.jet_pt[i], nom.jet_eta[i], nom.jet_phi[i], nom.jet_e[i])
            jet1 = LorentzVector() 
            jet1.SetPtEtaPhiE(nom.jet_pt[j], nom.jet_eta[j], nom.jet_phi[j], nom.jet_e[j])

            if withMatch:
                t = topDictFlat3l( jet0, jet1, lep0, lep1, lep2, met, nom.jet_tagWeightBin_DL1r_Continuous[i],
                                   nom.jet_tagWeightBin_DL1r_Continuous[j], isTop )
                combosTop['flatDicts'].append(t)
                t = topDictFourVec3l( jet0, jet1, lep0, lep1, lep2, met, nom.jet_tagWeightBin_DL1r_Continuous[i],    
                                      nom.jet_tagWeightBin_DL1r_Continuous[j], isTop )
                combosTop['fourVecDicts'].append(t)
            else:
                t = topDictFlat3l( jet0, jet1, lep0, lep1, lep2, met, nom.jet_tagWeightBin_DL1r_Continuous[i],
                                   nom.jet_tagWeightBin_DL1r_Continuous[j] )
                combosTop['flatDicts'].append(t)                                                                 
                t = topDictFourVec3l( jet0, jet1, lep0, lep1, lep2, met, nom.jet_tagWeightBin_DL1r_Continuous[i],
                                      nom.jet_tagWeightBin_DL1r_Continuous[j] )
                combosTop['fourVecDicts'].append(t)
                
    return combosTop 

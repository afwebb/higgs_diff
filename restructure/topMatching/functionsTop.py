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

def lorentzVecsTop(nom, topIdx0, topIdx1):
    '''                                                                                                                      
    Takes the indices of two jets identified to be bjets from top decay, return their LorentzVectors                  
    '''

    top0 = LorentzVector()
    top0.SetPtEtaPhiE(nom.jet_pt[topIdx0], nom.jet_eta[topIdx0], nom.jet_phi[topIdx0], nom.jet_e[topIdx0])            
    top1 = LorentzVector()
    top1.SetPtEtaPhiE(nom.jet_pt[topIdx1], nom.jet_eta[topIdx1], nom.jet_phi[topIdx1], nom.jet_e[topIdx1])

    return (top0, top1)

def lorentzVecsHiggs(nom, jetIdx0, jetIdx1, is3l, isF):                                                                
    '''                                                                                                               
    Higgs decays to two jets and one lepton, or two leptons. This returns lorentzVectors for each decay product candidate
    For H -> 2j, 1l case (not isF): returns jet0, jet1, met, lep0, lep1, (lep2 if is3l)                  
    For H -> 2l case (isF): Return met, lep0, lep1, lep2                                                      
    '''
                                                                                                                       
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met_met, 0, nom.met_phi, nom.met_met)
    
    lep0 = LorentzVector()
    lep0.SetPtEtaPhiE(nom.lep_Pt_0, nom.lep_Eta_0, nom.lep_Phi_0, nom.lep_E_0)
    
    lep1 = LorentzVector()
    lep1.SetPtEtaPhiE(nom.lep_Pt_1, nom.lep_Eta_1, nom.lep_Phi_1, nom.lep_E_1)                                               

    if is3l:                                                                                                                 
        lep2 = LorentzVector()
        lep2.SetPtEtaPhiE(nom.lep_Pt_2, nom.lep_Eta_2, nom.lep_Phi_2, nom.lep_E_2)
        
    if isF:
        return (met, lep0, lep1, lep2)
    else:
        jet0 = LorentzVector()
        jet0.SetPtEtaPhiE(nom.jet_pt[jetIdx0], nom.jet_eta[jetIdx0], nom.jet_phi[jetIdx0], nom.jet_e[jetIdx0])

        jet1 = LorentzVector()
        jet1.SetPtEtaPhiE(nom.jet_pt[jetIdx1], nom.jet_eta[jetIdx1], nom.jet_phi[jetIdx1], nom.jet_e[jetIdx1])            
        if is3l:
            return (jet0, jet1, met, lep0, lep1, lep2)
        else:
            return (jet0, jet1, met, lep0, lep1)

def jetCombosTop2lSS(nom, withMatch):
    ''' 
    Takes a ttree, returns an array of tuples of (dict, jet indices) for top2lSS matching for all possible combinations of jets 
    varType should be either flat or fourVec
    '''

    combosTop = {'flatDicts':[],'fourVecDicts':[],'jetIdx':[],'truthComb':[]}
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
            

            if withMatch:
                combosTop['flatDicts'].append( topDictFlat2lSS( nom, i, j, isTop) )
                combosTop['fourVecDicts'].append( topDictFourVec2lSS( nom, i, j, isTop) )
            else:
                combosTop['flatDicts'].append( topDictFlat2lSS( nom, i, j) )
                combosTop['fourVecDicts'].append( topDictFourVec2lSS( nom, i, j) )
            
    return combosTop #, truthComb

def jetCombosTop3l(nom, withMatch):
    '''                                                                                                                          Takes a ttree, returns an array of tuples of (dict, jet indices) for top2lSS matching for all possible combinations of 
    jets. withMatch option determines whether to include the "match" 
    '''

    combosTop = {'flatDicts':[],'fourVecDicts':[],'jetIdx':[],'truthComb':[]}                                     
    for i in range(len(nom.jet_pt)-1):
        for j in range(i+1, len(nom.jet_pt)):                                                                          
            combosTop['jetIdx'].append([i,j])
            isTop = 0                                                                                                
            if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthPartonLabel[i])==5:                                  
                if abs(nom.jet_parents[j])==6 and abs(nom.jet_truthPartonLabel[j])==5:
                    isTop = 1
                    combosTop['truthComb'] = [i,j]
                    
            if withMatch:                                                         
                combosTop['flatDicts'].append( topDictFlat3l( nom, i, j, isTop) )
                combosTop['fourVecDicts'].append( topDictFourVec3l( nom, i, j, isTop) )
            else:
                combosTop['flatDicts'].append( topDictFlat3l( nom, i, j) )
                combosTop['fourVecDicts'].append( topDictFourVec3l( nom, i, j) )
                
    return combosTop 

def combosHiggs2lSS(nom, withMatch):
    '''
    Take a ttree, return a dict with all possible combinations of higgs decay candidates. In the 2lSS case, 2 jets, 1 lepton.
    '''

    combosHiggs = {'higgsDicts':[],'jetIdx':[],'truthComb':[]}
    for l in range(2):
        for i in range(len(nom.jet_pt)-1):
            for j in range(i+1, len(nom.jet_pt)):
                combosHiggs['pairIdx'].append([l, i,j])
                isHiggs = 0                                                                                 
                if abs(nom.jet_parents[i])==25 and abs(nom.jet_parents[j])==25:
                    if (l == 0 and nom.lep_Parent_0==25) or (l == 1 and nom.lep_Parent_1 == 25):
                        isHiggs = 1
                        combosHiggs['truthComb'] = [l, i,j]

                if withMatch:
                    combosHiggs['higgsDicts'].append( higgsDict2lSS( nom, i, j, l, isHiggs) )
                else:
                    combosHiggs['higgsDicts'].append( higgsDict2lSS( nom, i, j, l) )

    return combosHiggs

def combosHiggs3lS(nom, withMatch):
    pass

def combosHiggs3lF(nom, withMatch):
    pass

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

def jetCombos3l(nom, withMatch):
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

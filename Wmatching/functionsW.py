'''
Functions for various object reconstruction or "match" MVAs
'''

import ROOT
from rootpy.vector import LorentzVector
from dictTop import topDict2lSS, topDict3l, topDictFourVec2lSS, topDictFourVec3l
from dictW import WTopDict2lSS, WTopDict3l, fourVecWTopDict2lSS, fourVecWTopDict3l
import pandas as pd
import numpy as np

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

def jetCombosTop(channel, nom, withMatch):
    ''' 
    Takes a ttree, returns an array of tuples of (dict, jet indices) for top2lSS matching for all combinations of jets 
    varType should be either flat or fourVec
    '''
    if channel=='2lSS': 
        flatDict = topDict2lSS
        fourVecDict = topDictFourVec2lSS
    elif channel=='3l':
        flatDict = topDict3l
        fourVecDict = topDictFourVec3l

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
                combosTop['flatDicts'].append( flatDict( nom, i, j, isTop) )
                combosTop['fourVecDicts'].append( fourVecDict( nom, i, j, isTop) )
            else:
                combosTop['flatDicts'].append( flatDict( nom, i, j) )
                combosTop['fourVecDicts'].append( fourVecDict( nom, i, j) )
            
    return combosTop

def WTopCombos(channel, nom, topIdx0, topIdx1, topScore, withMatch):
    '''                                                                                                                 
    Take a ttree, return a dict with all possible combinations of higgs decay candidates. In the 2lSS case, 2 jets, 1 lepton.
    Includes b-jets from tops                                                                                        
    '''

    #Set the channel, and corresponding dictionary. 2lSS - leptons run from 0 to 1, for 3l lepton can only be 1 or 2         
    if channel=='2lSS':                                                                                                
        lepRange=(0,1)                                                                                                      
        flatDict = WTopDict2lSS                                                                                       
    elif channel=='3l':                                                                                                 
        lepRange=(1,2)                                                                                                  
        flatDict = WTopDict3l
    else:                                                                                                                   
        print(f'Channel {channel} not recognized')
        return

    combosW = {'WTopDicts':[],'lepIdx':[],'truthComb':[]}

    for l in lepRange:                                                                                       
        combosW['lepIdx'].append([l])                                                                     
        if (l == 0 and abs(nom.lep_Parent_0)==24) or (l == 1 and abs(nom.lep_Parent_1)==24) or (l == 2 and abs(nom.lep_Parent_2) == 24):
            isW = 1                                                                 
            combosW['truthComb'] = [l]
        else:                                                                   
            isW = 0                                                                                     
        if withMatch:                                                                       
            combosW['WTopDicts'].append( flatDict( nom, l, topIdx0, topIdx1, topScore, isW) )
            #combosW['WTopDicts'].append( flatDict( nom, topIdx0, topIdx1, topScore, isW) ) 
        else:
            combosW['WTopDicts'].append( flatDict( nom, l, topIdx0, topIdx1, topScore) )
            #combosW['WTopDicts'].append( flatDict( nom, topIdx0, topIdx1, topScore) )

    return combosW

def findBestTopKeras(nom, channel, topModel, topNormFactors):
    '''
    Use a keras model to find the pair of jets most likely to be b-jets from the top decay. Return their indices
    if doTruth, also return the correct pairing based on truth info
    '''

    #load the appropriate model, get dicts for all possible pairings
    combosTop = jetCombosTop(channel, nom, 0)

    topDF = pd.DataFrame.from_dict(combosTop['flatDicts']) #convert dict to DF
    if len(list(topDF))<len(topNormFactors[1]):
        return 
    topDF=(topDF - topNormFactors[1])/(topNormFactors[0] - topNormFactors[1]) # Normalize DF
    topPred = topModel.predict(topDF.values) #feed pairings to the 
    topBest = np.argmax(topPred) # take the pairing with the highest score

    return {'bestComb':combosTop['jetIdx'][topBest], 'truthComb':combosTop['truthComb'], 'topScore':max(topPred)[0]}

def findBestWTop(nom, channel, model, normFactors, topIdx0, topIdx1, topScore):
    '''                                                                                                               
    Use a keras model to predict which physics objects in an event came from a Higgs decay                       
    '''

    combos = WTopCombos(channel, nom, topIdx0, topIdx1, topScore, 0)
    WDF = pd.DataFrame.from_dict(combos['WTopDicts'])

    #find combination of jets with highest W score
    WDF=(WDF - normFactors[1])/(normFactors[0]-normFactors[1])
    WPred = model.predict(WDF.values)                                                                    
    WBest = np.argmax(WPred)                                                                   
    
    return {'bestLep':combos['lepIdx'][WBest], 'truthLep':combos['truthComb'], 'WTopScore':max(WPred)[0]} 

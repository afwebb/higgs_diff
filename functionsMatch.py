'''
Functions for various object reconstruction or "match" MVAs
'''

import ROOT
from rootpy.vector import LorentzVector
from dictTop import topDict2lSS, topDictFourVec2lSS, topDict3l, topDictFourVec3l
from dictHiggs import higgsDict2lSS, higgsDict3lF, higgsDict3lS, higgsTopDict2lSS, higgsTopDict3lF, higgsTopDict3lS
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

    combosTop = {'flatDicts':[],'fourVecDicts':[],'jetIdx':[],'truthComb':[], 'higgsIdx':[]}
    for i in range(len(nom.jet_pt)-1):
        if nom.jet_jvt[i]<0.59: continue #Only include jets that pass JVT cut
        if nom.jet_parents[i]==25:
            combosTop['higgsIdx'].append(i)
        for j in range(i+1, len(nom.jet_pt)):
            combosTop['jetIdx'].append([i,j])
            if nom.jet_parents[j]==25:
                combosTop['higgsIdx'].append(j)
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

def higgsCombos(channel, nom, withMatch):
    '''
    For a given channel - 2lSS, 3lF or 3lS - return dict of all possible Higgs decay products, and which of those 
    is the correct pairing
    '''

    #Set the channel, and corresponding dictionary. 2lSS - leptons run from 0 to 1, for 3l lepton can only be 1 or 2
    if channel=='2lSS':
        lepRange=(0,1)
        flatDict = higgsDict2lSS
    elif channel=='3lS':
        lepRange=(1,2)
        flatDict = higgsDict3lS
    elif channel =='3lF':
        lepRange=(1,2)
        flatDict = higgsDict3lF
    else:
        print(f'Channel {channel} not recognized')
        return
    
    combosHiggs = {'higgsDicts':[],'pairIdx':[],'truthComb':[]} # initialize output

    if channel=='2lSS' or channel=='3lS': # for these channel, need 2 jets, 1 lepton
        for l in lepRange:
            for i in range(len(nom.jet_pt)-1):                                                                           
                for j in range(i+1, len(nom.jet_pt)): #loop over leptons and jet combos
                    combosHiggs['pairIdx'].append([l, i, j]) # Keep track of indices of this combo                      
                    isHiggs = 0                                  
                    #Only from Higgs if all three particles are matched to the Higgs
                    if abs(nom.jet_parents[i])==25 and abs(nom.jet_parents[j])==25:
                        if (l==0 and nom.lep_Parent_0==25) or (l==1 and nom.lep_Parent_1==25) or (l==2 and nom.lep_Parent_2==25):
                            isHiggs = 1   
                            combosHiggs['truthComb'] = [l, i, j] 
                    #Get the dict for this combo
                    if withMatch:                                                                                    
                        combosHiggs['higgsDicts'].append( flatDict( nom, i, j, l, isHiggs) )
                    else:
                        combosHiggs['higgsDicts'].append( flatDict( nom, i, j, l) )
    else:
        for l in lepRange: # Only two possibilities for 3lF - either lep 1 or lep 2 is form the Higgs
            combosHiggs['pairIdx'].append([l])
            if (l == 1 and nom.lep_Parent_1==25) or (l == 2 and nom.lep_Parent_2 == 25):
                isHiggs = 1
                combosHiggs['truthComb'] = [l]
            else:
                isHiggs = 0
            if withMatch:
                combosHiggs['higgsDicts'].append( flatDict( nom, l, isHiggs) )
            else:
                combosHiggs['higgsDicts'].append( flatDict( nom, l) )

    return combosHiggs    

def higgsTopCombos(channel, nom, topIdx0, topIdx1, topScore, withMatch):
    '''                                                                                                                 
    Take a ttree, return a dict with all possible combinations of higgs decay candidates. In the 2lSS case, 2 jets, 1 lepton.
    Includes b-jets from tops                                                                                        
    '''

    #Set the channel, and corresponding dictionary. 2lSS - leptons run from 0 to 1, for 3l lepton can only be 1 or 2         
    if channel=='2lSS':                                                                                                
        lepRange=(0,1)                                                                                                      
        flatDict = higgsTopDict2lSS                                                                                       
    elif channel=='3lS':                                                                                                 
        lepRange=(1,2)                                                                                                  
        flatDict = higgsTopDict3lS
    elif channel =='3lF':                                                                                                  
        lepRange=(1,2)
        flatDict = higgsTopDict3lF
    else:                                                                                                                   
        print(f'Channel {channel} not recognized')
        return

    combosHiggs = {'higgsDicts':[],'pairIdx':[],'truthComb':[]}

    if channel=='2lSS' or channel=='3lS':
        for l in lepRange:
            for i in range(len(nom.jet_pt)-1):
                for j in range(i+1, len(nom.jet_pt)):
                    combosHiggs['pairIdx'].append([l, i,j])
                    isHiggs = 0
                    if abs(nom.jet_parents[i])==25 and abs(nom.jet_parents[j])==25:
                        if (l==0 and nom.lep_Parent_0==25) or (l==1 and nom.lep_Parent_1==25) or (l==2 and nom.lep_Parent_2==25):
                            isHiggs = 1
                            combosHiggs['truthComb'] = [l, i, j]
                    if withMatch:                                                                                          
                        combosHiggs['higgsDicts'].append( flatDict( nom, i, j, l, topIdx0, topIdx1, topScore, isHiggs) )
                    else:                                                                                                 
                        combosHiggs['higgsDicts'].append( flatDict( nom, i, j, l, topIdx0, topIdx1, topScore) )
    else:                                                                                                             
        for l in lepRange:                                                                                       
            combosHiggs['pairIdx'].append([l])                                                                     
            if (l == 1 and nom.lep_Parent_1==25) or (l == 2 and nom.lep_Parent_2 == 25):
                isHiggs = 1                                                                 
                combosHiggs['truthComb'] = [l]
            else:                                                                   
                isHiggs = 0                                                                                     
            if withMatch:                                                                       
                combosHiggs['higgsDicts'].append( flatDict( nom, l, topIdx0, topIdx1, topScore, isHiggs) )
            else:
                combosHiggs['higgsDicts'].append( flatDict( nom, l, topIdx0, topIdx1, topScore) )

    return combosHiggs

def findBestTopKeras(nom, channel, topModel, topNormFactors):
    '''
    Use a keras model to find the pair of jets most likely to be b-jets from the top decay. Return their indices
    if doTruth, also return the correct pairing based on truth info
    '''

    #load the appropriate model, get dicts for all possible pairings
    combosTop = jetCombosTop(channel, nom, 0)

    topDF = pd.DataFrame.from_dict(combosTop['flatDicts']) #convert dict to DF
    topDF=(topDF - topNormFactors[1])/(topNormFactors[0] - topNormFactors[1]) # Normalize DF
    topPred = topModel.predict(topDF.values) #feed pairings to the 
    topBest = np.argmax(topPred) # take the pairing with the highest score

    return {'bestComb':combosTop['jetIdx'][topBest], 'truthComb':combosTop['truthComb'], 'topScore':max(topPred)[0], 'higgsIdx':combosTop['higgsIdx']}

def findBestHiggs(nom, channel, model, normFactors):
    '''
    Use a keras model to predict which physics objects in an event came from a Higgs decay
    '''

    combos = higgsCombos(channel, nom, 0)

    higgsDF = pd.DataFrame.from_dict(combos['higgsDicts'])

    #find combination of jets with highest higgs score                                                                       
    higgsDF=(higgsDF - normFactors[1])/(normFactors[0]-normFactors[1])
    higgsPred = model.predict(higgsDF.values)      
    higgsBest = np.argmax(higgsPred)  

    return {'bestComb':combos['pairIdx'][higgsBest], 'truthComb':combos['truthComb'], 'higgsScore':max(higgsPred)[0]}

def findBestHiggsTop(nom, channel, model, normFactors, topIdx0, topIdx1, topScore):
    '''                                                                                                               
    Use a keras model to predict which physics objects in an event came from a Higgs decay                       
    '''

    combos = higgsTopCombos(channel, nom, topIdx0, topIdx1, topScore, 0)
    higgsDF = pd.DataFrame.from_dict(combos['higgsDicts'])

    #find combination of jets with highest higgs score
    higgsDF=(higgsDF - normFactors[1])/(normFactors[0]-normFactors[1])
    higgsPred = model.predict(higgsDF.values)                                                                    
    higgsBest = np.argmax(higgsPred)                                                                   
    
    return {'bestComb':combos['pairIdx'][higgsBest], 'truthComb':combos['truthComb'], 'higgsTopScore':max(higgsPred)[0]} 

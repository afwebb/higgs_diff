# Defines dictionaries to be used for higgs reconstruction algorithms, which try to identify the decay products of the Higgs

import ROOT
from rootpy.vector import LorentzVector

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

def lorentzVecsTop(nom, topIdx0, topIdx1):
    '''
    Takes the indices of two jets identified to be bjets from top decay, return their LorentzVectors 
    '''

    top0 = LorentzVector()                                                                                                   
    top0.SetPtEtaPhiE(nom.jet_pt[topIdx0], nom.jet_eta[topIdx0], nom.jet_phi[topIdx0], nom.jet_e[topIdx0])
    top1 = LorentzVector()                                                                                         
    top1.SetPtEtaPhiE(nom.jet_pt[topIdx1], nom.jet_eta[topIdx1], nom.jet_phi[topIdx1], nom.jet_e[topIdx1])

    return (top0, top1)

def higgsDict2lSS(nom, jetIdx0, jetIdx1, lepIdx, match=-1):#, lep, met, jet1_MV2c10, jet2_MV2c10, lepO, jet1_numTrk, jet2_numTrk, match=-1):
    '''
    Takes a ttree, two jet indices, and which of the two leptons is considered part of the Higgs candidate. 
    Returns a dict for identifying Higgs decay products
    Give match if using for training. =1 if these jets, leptons are truth matched to the Higgs, 0 if not
    '''

    #Initialize four-vectors of decay product candidates
    jet0, jet1, met, lep0, lep1 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 0, 0)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs
    if lepIdx == 0:
        lepH = lep0
        lepT = lep1
    elif lepIdx == 1:
        lepH = lep1
        lepT = lep0
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 0 or 1")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['dRjj'] = jet0.DeltaR(jet1)
    k['Ptjj'] = (jet0+jet1).Pt()
    k['Mjj'] = (jet0+jet1).M()

    k['dRlj0'] = lepH.DeltaR(jet0)
    k['Ptlj0'] = (lepH+jet0).Pt()
    k['Mlj0'] = (lepH+jet0).M()

    k['dRlj1'] = lepH.DeltaR(jet1)
    k['Ptlj1'] = (lepH+jet1).Pt()
    k['Mlj1'] = (lepH+jet1).M()

    k['dR(jj)(l)'] = (jet0 + jet1).DeltaR(lepH)
    k['MhiggsCand'] = (jet0+jet1+lepH).M()

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    k['lep_Pt_T'] = lepT.Pt()

    k['dR(jj)(lepT)'] = (jet0+jet1).DeltaR(lepT)
    k['Mj0lO'] = (jet0+lepT).M()
    k['Mj1lO'] = (jet1+lepT).M()

    return k

def higgsDict3lF(nom, lepIdx, match=-1):
    '''                                                                                                                      
    Takes a ttree, index of which of the leptons is considered part of the Higgs candidate. Must be 1 or 2 (lep0 is from top)
    Returns a dict for identifying Higgs decay products                     
    Gives match if using for training. =1 if these jets, leptons are truth matched to the Higgs, 0 if not                  
    ''' 

    k = {}
    met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, 0, 0, 1, 1)

    #Assign leptons to parents
    lepT0 = lep0
    if lepIdx == 1:
        lepH, lepT1 = lep1, lep2
    elif lepIdx == 2:
        lepH, lepT1, = lep2, lep1
    else:                                                                                                             
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0

    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['lep_Pt_T0'] = lepT0.Pt()
    k['lep_Pt_T1'] = lepT1.Pt()

    k['MllHT0)'] = (lepH+lepT0).M()
    k['MllHT1'] = (lepH+lepT1).M()
    k['MllT0T1)'] = (lepT0+lepT1).M()
    
    k['PtllHT0'] = (lepH+lepT0).Pt()
    k['PtllHT1'] = (lepH+lepT1).Pt()

    k['dRllHT0'] = lepH.DeltaR(lepT0)
    k['dRllHT1'] = lepH.DeltaR(lepT1)
    k['dRllT0T1'] = lepT0.DeltaR(lepT1)

    k['dRl0Met'] = lepH.DeltaR(met)
    k['dRl1Met'] = lepT0.DeltaR(met)
    k['dRl2Met'] = lepT1.DeltaR(met)

    k['dRllHT0Met'] = (lepH+lepT0).DeltaR(met)
    k['dRllHT1Met'] = (lepH+lepT1).DeltaR(met)
    k['dRllT0T1Met'] = (lepT0+lepT1).DeltaR(met)

    k['MllHT0Met'] = (lepH+lepT0+met).M()
    k['MllHT1Met'] = (lepH+lepT1+met).M()
    k['MllT0T1Met'] = (lepT0+lepT1+met).M()

    k['met'] = met.Pt()

    return k

def higgsDict3lS(nom, jetIdx0, jetIdx1, lepIdx, match=-1):
    '''                                                                                             
    Takes a ttree, two jet indices, and which of the three leptons is considered part of the Higgs candidate (1 or 2).    
    Returns a dict for identifying Higgs decay products                                                     
    Give match if using for training. =1 if these jets, leptons are truth matched to the Higgs, 0 if not        
    '''

    #Initialize four-vectors of decay product candidates                                                                
    jet0, jet1, met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 1, 0)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs     
    if lepIdx == 1:                                                                      
        lepH = lep1
        lepT0 = lep0                                                                                                    
        lepT1 = lep2
    elif lepIdx == 2:                                                                                           
        lepH = lep2                                                                                                 
        lepT0 = lep0
        lepT1 = lep1
    else:                                                                                                         
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['dR(j)(j)'] = jet0.DeltaR(jet1)
    k['M(jj)'] = (jet0+jet1).M()

    k['dR(l)(j0)'] = lepH.DeltaR(jet0)
    k['dR(l)(j1)'] = lepH.DeltaR(jet1)

    k['dR(jj)(l)'] = (jet0 + jet1).DeltaR(lepH)
    k['M(jjl)'] = (jet0+jet1+lepH).M()

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    k['lep_Pt_T0'] = lepT0.Pt()
    k['dR(jj)(lepT0)'] = (jet0+jet1).DeltaR(lepT0)
    k['M(jjlO1)'] = (jet0+jet1+lepT0).M()

    k['lep_Pt_O2'] = lepT1.Pt()
    k['dR(jj)(lepT1)'] = (jet0+jet1).DeltaR(lepT1)
    k['M(jjlO2)'] = (jet0+jet1+lepT1).M()

    k['dR(lO1)(lO2)'] = lepT0.DeltaR(lepT1)

    k['dR(l)(lO1)'] = lepH.DeltaR(lepT0)
    k['dR(l)(lO2)'] = lepH.DeltaR(lepT1)

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    return k

def higgsTopDict2lSS(nom, jetIdx0, jetIdx1, lepIdx, topIdx0, topIdx1, match=-1):
    '''
    
    '''

    jet0, jet1, met, lep0, lep1 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 0, 0)
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs
    if lepIdx == 0:                                                                                                          
        lepH = lep0                                                                                                     
        lepT = lep1                                                                                                 
    elif lepIdx == 1:                                                                                                 
        lepH = lep1                                                                                              
        lepT = lep0                                                                                        
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 0 or 1")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt'] = lepH.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()
    
    #k['lep_Pt_T'] = lepT.Pt()

    k['dRjj'] = jet0.DeltaR(jet1)
    #k['Ptjj'] = (jet0+jet1).Pt()
    k['Mjj'] = (jet0+jet1).M()

    k['dRlj0'] = lepH.DeltaR(jet0)
    #k['Ptlj0'] = (lepH+jet0).Pt()
    k['Mlj0'] = (lepH+jet0).M()
    k['dRlj1'] = lepH.DeltaR(jet1)
    #k['Ptlj1'] = (lepH+jet1).Pt()
    k['Mlj1'] = (lepH+jet1).M()

    k['dRlt0'] = lepH.DeltaR(top0)
    #k['Ptlt0'] = (lepH+top0).Pt()
    k['Mlt0'] = (lepH+top0).M()

    k['dRlt1'] = lepH.DeltaR(top1)
    #k['Ptlj1'] = (lepH+top1).Pt()
    k['Mlt1'] = (lepH+top1).M()

    k['dRjt00'] = jet0.DeltaR(top0)
    #k['Ptjt00'] = (jet0+top0).Pt()
    #k['Mjt00'] = (jet0+top0).M()

    k['dRjt01'] = jet0.DeltaR(top1)
    #k['Ptjt01'] = (jet0+top1).Pt()
    #k['Mjt01'] = (jet0+top1).M()

    k['dRjt10'] = jet1.DeltaR(top0)
    #k['Ptjt10'] = (jet1+top0).Pt()
    #k['Mljt10'] = (jet1+top0).M()

    k['dRjt11'] = jet1.DeltaR(top1)
    #k['Ptjt11'] = (jet1+top1).Pt()
    #k['Mljt11'] = (jet1+top1).M()

    #k['Mttl'] = (top0+top1+lepH).M()

    k['dR(jj)(lepT)'] = (jet0+jet1).DeltaR(lepT)

    #k['MtlT0'] = (top0+lepT).M()
    #k['MtlT1'] = (top1+lepT).M()

    k['dRtlT0'] = top0.DeltaR(lepT)
    k['dRtlT1'] = top1.DeltaR(lepT)

    k['dR(jj)(l)'] = (jet0 + jet1).DeltaR(lepH + met)
    k['MhiggsCand'] = (jet0+jet1+lepH).M()

    higgsCand = jet0+jet1+lepH

    k['dRht0'] = higgsCand.DeltaR(top0)
    k['dRht1'] = higgsCand.DeltaR(top1)

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    return k

def higgsTopDict3lF(nom, lepIdx, topIdx0, topIdx1, match=-1):
    '''
    '''

    met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, 0, 0, 1, 1)
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)

    k = {}
    if match!=-1:                                                                                                            
        k['match'] = match

    return k

def higgsTopDict3lS(nom, jet0, jet1, lepIdx, topIdx0, topIdx1, match =-1):
    '''
    '''
    
    jet0, jet1, met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 1, 0)
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)
    
    k = {}
    if match!=-1:                                                                                                       
        k['match'] = match

    return k

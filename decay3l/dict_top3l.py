def topDict(jet1, jet2, lep1, lep2, lep3, met, jet1_MV2c10, jet2_MV2c10, jet1_numTrk, jet2_numTrk, match=-1):
#def topDict(jet1, jet2, lep1, lep2, lep3, met, jet1_MV2c10, jet2_MV2c10, jet1_jvt, jet2_jvt, jet1_numTrk, jet2_numTrk, match=-1):
    k = {}

    if match!=-1:
        k['match'] = match

    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    #k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj00'] = lep1.DeltaR(jet1)
    k['Mlj00'] = (lep1+jet1).M()

    k['dRlj01'] = lep1.DeltaR(jet2)
    k['Mlj01'] = (lep1+jet2).M()

    k['dRlj10'] = lep2.DeltaR(jet1)
    k['Mlj10'] = (lep2+jet1).M()

    k['dRlj11'] = lep2.DeltaR(jet2)
    k['Mlj11'] = (lep2+jet2).M()

    k['dRlj20'] = lep3.DeltaR(jet1)
    k['Mlj20'] = (lep3+jet1).M()

    k['dRlj21'] = lep3.DeltaR(jet2)
    k['Mlj21'] = (lep3+jet2).M()

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    k['dRj0met'] = jet1.DeltaR(met)
    k['dRj1met'] = jet2.DeltaR(met)

    '''
    if jet1_jvt>0.59:
        k['jet_jvt_0'] = 1
    else:
        k['jet_jvt_0'] = 0

    if jet2_jvt>0.59:
        k['jet_jvt_1'] = 1
    else:
        k['jet_jvt_1'] = 0
    '''
    k['jet_numTrk_0'] =jet1_numTrk
    k['jet_numTrk_1'] =jet2_numTrk

    #k['met'] = met.Pt()
    #k['met_phi']= met.Phi()

    return k

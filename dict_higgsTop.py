def higgsTopDict(jet1, jet2, lep, met, jet1_MV2c10, jet2_MV2c10, topJet1, topJet2, lepO, jet1_jvt, jet2_jvt, jet1_numTrk, jet2_numTrk, match=-1):
    k = {}

    if match!=-1:
        k['match'] = match

    k['lep_Pt'] = lep.Pt()
    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()
    
    #k['lep_Pt_Other'] = lepO.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj0'] = lep.DeltaR(jet1)
    k['Ptlj0'] = (lep+jet1).Pt()
    k['Mlj0'] = (lep+jet1).M()
    k['dRlj1'] = lep.DeltaR(jet2)
    k['Ptlj1'] = (lep+jet2).Pt()
    k['Mlj1'] = (lep+jet2).M()

    k['dRlt0'] = lep.DeltaR(topJet1)
    k['Ptlt0'] = (lep+topJet1).Pt()
    k['Mlt0'] = (lep+topJet1).M()

    k['dRlt1'] = lep.DeltaR(topJet2)
    k['Ptlj1'] = (lep+topJet2).Pt()
    k['Mlj1'] = (lep+topJet2).M()

    k['dRjt00'] = jet1.DeltaR(topJet1)
    k['Ptjt00'] = (jet1+topJet1).Pt()
    k['Mljt00'] = (jet1+topJet1).M()

    k['dRjt01'] = jet1.DeltaR(topJet2)
    k['Ptjt01'] = (jet1+topJet2).Pt()
    k['Mljt01'] = (jet1+topJet2).M()

    k['dRjt10'] = jet2.DeltaR(topJet1)
    k['Ptjt10'] = (jet2+topJet1).Pt()
    k['Mljt10'] = (jet2+topJet1).M()

    k['dRjt11'] = jet2.DeltaR(topJet2)
    k['Ptjt11'] = (jet2+topJet2).Pt()
    k['Mljt11'] = (jet2+topJet2).M()

    k['Mttl'] = (topJet1+topJet2+lep).M()

    k['dR(jj)(lepOther)'] = (jet1+jet2).DeltaR(lepO)

    k['MtlOther0'] = (topJet1+lepO).M()
    k['MtlOther1'] = (topJet2+lepO).M()

    k['dRtlOther0'] = topJet1.DeltaR(lepO)
    k['dRtlOther1'] = topJet2.DeltaR(lepO)

    k['dR(jj)(l)'] = (jet1 + jet2).DeltaR(lep + met)
    k['MhiggsCand'] = (jet1+jet2+lep).M()

    higgsCand = jet1+jet2+lep

    k['dRht0'] = higgsCand.DeltaR(topJet1)
    k['dRht1'] = higgsCand.DeltaR(topJet2)

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    k['jet_jvt_0'] = jet1_jvt
    k['jet_jvt_1'] = jet2_jvt

    k['jet_numTrk_0'] = jet1_numTrk
    k['jet_numTrk_1'] = jet2_numTrk

    return k

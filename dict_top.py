def topDict(jet1, jet2, lep1, lep2, met, jet1_MV2c10, jet2_MV2c10, match=-1):
    k = {}

    if match!=-1:
        k['match'] = match

    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj00'] = lep1.DeltaR(jet1)
    k['Mlj00'] = (lep1+jet1).M()

    k['dRlj01'] = lep1.DeltaR(jet2)
    k['Mlj01'] = (lep1+jet2).M()

    k['dRlj10'] = lep2.DeltaR(jet1)
    k['Mlj10'] = (lep2+jet1).M()

    k['dRlj11'] = lep2.DeltaR(jet2)
    k['Mlj11'] = (lep2+jet2).M()

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    k['dRjjl0'] = (jet1+jet2).DeltaR(lep1)
    k['dRjjl1'] = (jet1+jet2).DeltaR(lep2)

    k['Mjjl0'] = (jet1+jet2+lep1).M()
    k['Mjjl1'] = (jet1+jet2+lep2).M()

    return k

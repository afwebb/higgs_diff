def higgsDict(jet1, jet2, lep, met, jet1_MV2c10, jet2_MV2c10, lepO, match=-1):
    k = {}

    if match!=-1:
        k['match'] = match

    k['lep_Pt'] = lep.Pt()
    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj0'] = lep.DeltaR(jet1)
    k['Ptlj0'] = (lep+jet1).Pt()
    k['Mlj0'] = (lep+jet1).M()

    k['dRlj1'] = lep.DeltaR(jet2)
    k['Ptlj1'] = (lep+jet2).Pt()
    k['Mlj1'] = (lep+jet2).M()

    k['dR(jj)(l)'] = (jet1 + jet2).DeltaR(lep)
    k['MhiggsCand'] = (jet1+jet2+lep).M()

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    k['lep_Pt_Other'] = lepO.Pt()

    k['dR(jj)(lepOther)'] = (jet1+jet2).DeltaR(lepO)
    k['Mj0lO'] = (jet1+lepO).M()
    k['Mj1lO'] = (jet2+lepO).M()

    return k

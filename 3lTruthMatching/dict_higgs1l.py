def higgs1lDict(jet1, jet2, lep, met, jet1_MV2c10, jet2_MV2c10, lepO1, lepO2, jet1_numTrk, jet2_numTrk, match=-1):

    k = {}

    if match!=-1:
        k['match'] = match

    k['lep_Pt'] = lep.Pt()
    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dR(j)(j)'] = jet1.DeltaR(jet2)
    k['M(jj)'] = (jet1+jet2).M()

    k['dR(l)(j0)'] = lep.DeltaR(jet1)
    k['dR(l)(j1)'] = lep.DeltaR(jet2)

    k['dR(jj)(l)'] = (jet1 + jet2).DeltaR(lep)
    k['M(jjl)'] = (jet1+jet2+lep).M()

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    k['lep_Pt_O1'] = lepO1.Pt()
    k['dR(jj)(lepO1)'] = (jet1+jet2).DeltaR(lepO1)
    k['M(jjlO1)'] = (jet1+jet2+lepO1).M()

    k['lep_Pt_O2'] = lepO2.Pt()
    k['dR(jj)(lepO2)'] = (jet1+jet2).DeltaR(lepO2)
    k['M(jjlO2)'] = (jet1+jet2+lepO2).M()

    k['dR(lO1)(lO2)'] = lepO1.DeltaR(lepO2)

    k['dR(l)(lO1)'] = lep.DeltaR(lepO1)
    k['dR(l)(lO2)'] = lep.DeltaR(lepO2)

    k['jet_numTrk_0'] = jet1_numTrk
    k['jet_numTrk_1'] = jet2_numTrk

    return k

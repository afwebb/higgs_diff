def higgs1lDict(jet1, jet2, lep, met, jet1_MV2c10, jet2_MV2c10, lepO1, lepO2, jet1_jvt, jet2_jvt, jet1_numTrk, jet2_numTrk, match=-1):

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

    k['lep_Pt_Other1'] = lepO1.Pt()

    k['dR(jj)(lepOther1)'] = (jet1+jet2).DeltaR(lepO1)
    k['MjjlO1'] = (jet1+jet2+lepO1).M()

    k['lep_Pt_Other2'] = lepO2.Pt()

    k['dR(jj)(lepOther2)'] = (jet1+jet2).DeltaR(lepO2)
    k['MjjlO2'] = (jet1+jet2+lepO2).M()

    k['dRllO1O2'] = lepO1.DeltaR(lepO2)

    k['dRllHO1'] = lep.DeltaR(lepO1)
    k['dRllHO2'] = lep.DeltaR(lepO2)

    k['jet_jvt_0'] = jet1_jvt
    k['jet_jvt_1'] = jet1_jvt

    k['jet_numTrk_0'] = jet1_numTrk
    k['jet_numTrk_1'] = jet1_numTrk

    return k

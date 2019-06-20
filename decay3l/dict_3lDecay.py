def decayDict(lep0, lep1, lep2, met, top0, top1, decay=-1):
    k = {}

    if decay!=-1:
        k['decay'] = decay

    k['lep_Pt_0'] = lep0.Pt()
    k['lep_Pt_1'] = lep1.Pt()
    k['lep_Pt_2'] = lep2.Pt()
    
    k['dRll01'] = lep0.DeltaR(lep1)
    k['dRll02'] = lep0.DeltaR(lep2)
    k['dRll12'] = lep1.DeltaR(lep2)

    k['Mll01'] = (lep0+lep1).M()
    k['Mll02'] = (lep0+lep2).M()
    k['Mll12'] = (lep1+lep2).M()
    k['Mlll'] = (lep0+lep1+lep2).M()

    k['top_Pt_0'] = top0.Pt()
    k['top_Pt_1'] = top1.Pt()

    k['dRlt00'] = lep0.DeltaR(top0)
    k['dRlt01'] = lep0.DeltaR(top1)

    k['dRlt10'] = lep1.DeltaR(top0)
    k['dRlt11'] = lep1.DeltaR(top1)

    k['dRlt20'] = lep2.DeltaR(top0)
    k['dRlt21'] = lep2.DeltaR(top1)

    k['Mlt00'] = (lep0+top0).M()
    k['Mlt01'] = (lep0+top1).M()

    k['Mlt10'] = (lep1+top0).M()
    k['Mlt11'] = (lep1+top1).M()

    k['Mlt20'] = (lep2+top0).M()
    k['Mlt21'] = (lep2+top1).M()

    k['met'] = met.Pt()
    k['met_phi'] = met.Phi()

    return k

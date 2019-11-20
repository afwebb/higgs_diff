def higgs2lDict(lep0, lep1, lep2, met, top0, top1, match=-1):

    k = {}

    if match!=-1:
        k['match'] = match

    k['lep_Pt_0'] = lep0.Pt()
    k['lep_Pt_1'] = lep1.Pt()
    k['lep_Pt_2'] = lep2.Pt()

    k['Mll01'] = (lep0+lep1).M()
    k['Mll02'] = (lep0+lep2).M()
    k['Mll12'] = (lep1+lep2).M()
    
    k['Ptll01'] = (lep0+lep1).Pt()
    k['Ptll02'] = (lep0+lep2).Pt()

    k['dRll01'] = lep0.DeltaR(lep1)
    k['dRll02'] = lep0.DeltaR(lep2)
    k['dRll12'] = lep1.DeltaR(lep2)

    k['dRl0Met'] = lep0.DeltaR(met)
    k['dRl1Met'] = lep1.DeltaR(met)
    k['dRl2Met'] = lep2.DeltaR(met)

    k['dRll01Met'] = (lep0+lep1).DeltaR(met)
    k['dRll02Met'] = (lep0+lep2).DeltaR(met)
    k['dRll12Met'] = (lep1+lep2).DeltaR(met)

    k['Mll01Met'] = (lep0+lep1+met).M()
    k['Mll02Met'] = (lep0+lep2+met).M()
    k['Mll12Met'] = (lep1+lep2+met).M()

    k['dRllm01t0'] = (lep0+lep1+met).DeltaR(top0)
    k['dRllm01t1'] = (lep0+lep1+met).DeltaR(top1)

    k['dRllm02t0'] = (lep0+lep2+met).DeltaR(top0)
    k['dRllm02t1'] = (lep0+lep2+met).DeltaR(top1)

    k['dRlt10'] = lep1.DeltaR(top0)
    k['dRlt20'] = lep2.DeltaR(top0)

    k['dRlt10'] = lep1.DeltaR(top1)
    k['dRlt20'] = lep2.DeltaR(top1)

    k['met'] = met.Pt()

    return k

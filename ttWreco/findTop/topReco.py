import uproot4
import numpy as np
import pandas as pd
import sys
import awkward1 as ak
import numba as nb
import random
import sklearn as sk

f = uproot4.open(sys.argv[1])
nom = f['nominal']

#lepton branches to add to data frame
branches = ['lep_Pt_0', 'lep_Eta_0', 'lep_Phi_0', #'lep_Parent_0',
            'lep_Pt_1', 'lep_Eta_1', 'lep_Phi_1', #'lep_Parent_1',
            'met_met', 'met_phi', 'nJets_OR', 'nJets_OR_DL1r_70',
        ]

jetBranches = ["jet_pt","jet_DL1r","jet_truthflav","jet_eta","jet_phi",'jet_parents']
jetKin = ['jet_pt','jet_eta','jet_phi']

awkArr = nom.arrays(branches+jetBranches,library='ak', how='zip')
awkArr = awkArr[ak.num(awkArr.jet)>3]
awkArr = awkArr[ak.num(awkArr.jet[awkArr.jet.truthflav == 5])==2]
print(awkArr)

lepDF = pd.DataFrame()
for b in branches:
    lepDF[b] = ak.to_list(awkArr[b])

bjets = awkArr.jet[awkArr.jet.truthflav == 5]
notBjets = awkArr.jet[awkArr.jet.truthflav != 5]

def truth_tops(bjets, lepDF):
    '''Add kinematics of b-jets to dict'''
    print('Adding truth bjets')
    df = pd.DataFrame()
    
    #Convert jet awk arrays to dataframe
    for l in range(2):
        for b in ['pt','eta','phi', 'DL1r']:
            df['jet_'+b+'_'+str(l)] = [j[b][l] for j in bjets]

    #df['jet_pt_0'] = [j.pt[0] for j in bjets]
    #df['jet_eta_0'] = [j.eta[0] for j in bjets]
    #df['jet_phi_0'] = [j.phi[0] for j in bjets]
    #df['jet_DL1r_0'] = [j.DL1r[0] for j in bjets]
    
    #df['jet_pt_1'] = [j.pt[1] for j in bjets]
    #df['jet_eta_1'] = [j.eta[1] for j in bjets]
    #df['jet_phi_1'] = [j.phi[1] for j in bjets]
    #df['jet_DL1r_1'] = [j.DL1r[1] for j in bjets]

    #add other branches to dataframe
    df = pd.concat([df, lepDF], axis=1)

    df['match'] = 1

    return df

def bad_tops(notBjets, lepDF):
    '''Add kinematics of non-bjets to the dict'''
    print("Adding bad jets")
    numJets = ak.num(notBjets)
    
    totDF = pd.DataFrame()

    for i in range(3):
        badJets = [np.random.choice(x, (2,1), replace=False) for x in numJets]
        df = pd.DataFrame()
        #Convert jet awk arrays to dataframe                                                                                 

        for l in range(2):
            for b in ['pt','eta','phi', 'DL1r']:
                df['jet_'+b+'_'+str(l)] = [j[b][i[l]][0] for i, j in zip(badJets, notBjets)]
         
        #add other branches to dataframe                                                      
        df = pd.concat([df, lepDF], axis=1)
        totDF = pd.concat([totDF, df])
        
    totDF['match'] = 0

    return totDF

def partial_tops(bjets, notBjets, lepDF):
    '''Add kinematics of non-bjets to the dict'''
    print("Adding bad jets")

    totDF = pd.DataFrame()

    numJets = ak.num(notBjets)

    for i in range(2):
        df = pd.DataFrame()

        badJets = [random.randint(0, x-1) for x in numJets]
        goodJets = [random.randint(0,1) for x in range(len(numJets))]

        #Convert jet awk arrays to dataframe                                                    
        for b in ['pt','eta','phi', 'DL1r']:         
            df['jet_'+b+'_0'] = [j[b][i] for i, j in zip(goodJets, bjets)]
            df['jet_'+b+'_1'] = [j[b][i] for i, j in zip(badJets, notBjets)]
        
        #df['jet_pt_0'] = [j.pt[i] for i, j in zip(goodJets, notBjets)]
        #df['jet_eta_0'] = [j.eta[i] for i, j in zip(goodJets, notBjets)]
        #df['jet_phi_0'] = [j.phi[i] for i, j in zip(goodJets, notBjets)]

        #df['jet_pt_1'] = [j.pt[i] for i, j in zip(badJets, notBjets)]
        #df['jet_eta_1'] = [j.eta[i] for i, j in zip(badJets, notBjets)]
        #df['jet_phi_1'] = [j.phi[i] for i, j in zip(badJets, notBjets)]

        #add other branches to dataframe                                                                                                      
        df = pd.concat([df, lepDF], axis=1)
        totDF = pd.concat([totDF, df])

    totDF['match'] = 0
    return totDF

goodDF = truth_tops(bjets, lepDF)
badDF = bad_tops(notBjets, lepDF)
partDF = partial_tops(bjets, notBjets, lepDF)
outDF = pd.concat([goodDF, badDF, partDF])
outDF = sk.utils.shuffle(outDF)

#Write output
outFile = sys.argv[1].replace('.root','.csv')
outFile = 'csvFiles/'+'/'.join(outFile.split('/')[-2:])
outDF.to_csv(outFile, index=False)

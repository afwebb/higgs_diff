import uproot4
import numpy as np
import pandas as pd
import sys
import awkward1 as ak
import numba as nb

f = uproot4.open(sys.argv[1])
nom = f['nominal']

df = pd.DataFrame()

#Branches to add to data frame
branches = ['lep_Pt_0', 'lep_Eta_0', 'lep_Phi_0', 'lep_Parent_0',
            'lep_Pt_1', 'lep_Eta_1', 'lep_Phi_1', 'lep_Parent_1',
            'met_met', 'met_phi', 'nJets_OR', 'nJets_OR_DL1r_70',
        ]

#Add top highest b-tagged jets to dataframe
awkArr = nom.arrays(["jet_pt","jet_DL1r","jet_eta","jet_phi",'jet_parents'],library='ak', how='zip')

j1,j2 = [],[]
for j in ak.to_list(awkArr.jet.DL1r):
    jIdx = ak.argmax(j)
    j1.append(jIdx)
    j2.append(ak.argmax(j[:jIdx]+j[jIdx+1:]))

#Convert jet awk arrays to dataframe
df['jet_Pt_0'] = [j.jet.pt[int(i)] for j, i in zip(awkArr, j1)]
df['jet_Eta_0'] = [j.jet.eta[int(i)] for j, i in zip(awkArr, j1)]
df['jet_Phi_0'] = [j.jet.phi[int(i)] for j, i in zip(awkArr, j1)]

df['jet_Pt_1'] = [j.jet.pt[int(i)] for j, i in zip(awkArr, j2)]
df['jet_Eta_1'] = [j.jet.eta[int(i)] for j, i in zip(awkArr, j2)]
df['jet_Phi_1'] = [j.jet.phi[int(i)] for j, i in zip(awkArr, j2)]

#add other branches to dataframe
for b in branches:
    df[b] = nom[b].array(library='pd')

#Remove events where leptons are not truth matched
df = df[(abs(df['lep_Parent_0'])==6) | (abs(df['lep_Parent_0'])==24)]
df = df[(abs(df['lep_Parent_1'])==6) | (abs(df['lep_Parent_1'])==24)]

#convert lep parent to fromW
truthId = {6:0, -6:0, 24:1, -24:1}
df['fromW'] = df['lep_Parent_0'].map(truthId)
#df = df.drop('lep_Parent_0', axis=1)
#df = df.drop('lep_Parent_1', axis=1)

#Write output
outFile = sys.argv[1].replace('.root','.csv')
outFile = 'csvFiles/'+'/'.join(outFile.split('/')[-2:])
df.to_csv(outFile)
import pandas as pd
import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]

inDF = pd.read_csv(inputFile)

#inDF = inDF[inDF['comboScore'] > 0.2]

dropCol = ['top_E_0','top_E_1','top_Eta_0','top_Eta_1','top_MV2c10_0','top_MV2c10_1','top_Phi_0','top_Phi_1','top_Pt_0','top_Pt_1']

for d in dropCol:
    inDF = inDF.drop(d, axis=1)

inDF.to_csv(outputFile, index=False)

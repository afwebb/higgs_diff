import pandas as pd
import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]

inDF = pd.read_csv(inputFile)

inDF = inDF[inDF['comboScore'] > 0.3]
inDF.drop('comboScore', axis=1)

inDF = inDF.drop('jet_Pt_b0', axis=1)
inDF = inDF.drop('jet_Eta_b0', axis=1)
inDF = inDF.drop('jet_Phi_b0', axis=1)
inDF = inDF.drop('jet_E_b0', axis=1)

inDF = inDF.drop('jet_Pt_b1', axis=1)
inDF = inDF.drop('jet_Eta_b1', axis=1)
inDF = inDF.drop('jet_Phi_b1', axis=1)
inDF = inDF.drop('jet_E_b1', axis=1)

inDF.to_csv(outputFile, index=False)

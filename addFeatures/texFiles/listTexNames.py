'''
Plot input features of a dataset, correlation matrix
Save as .pdf, and write latex file with plots included in plots/<out string> directory
Usage: python3.6 featurePlots.py <input dataset.csv> <out string>
'''

import pandas
import xgboost as xgb
import pickle
import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from namePlot import name
#get_ipython().run_line_magic('matplotlib', 'inline')

inFile = sys.argv[1]
outDir = sys.argv[2]

inDF = pd.read_csv(inFile, nrows=1e2)
inDF = inDF.dropna()

texfile = open('tabNames_'+outDir+".tex", "w")

icount = 1

nameList = []

for c in inDF:
    print(f'"{c}"')
    if c=='match': continue
    xName, r = name(c)
    xName = xName.replace(" [GeV]", '')
    nameList.append(xName)

print(len(nameList), len(nameList)//3)
print("\\begin{table}[h!]", file=texfile)
print("  \\begin{center}", file=texfile)
print("  \\begin{tabular}{ccc}", file=texfile)
for i in range(len(nameList)//3):
    print(f"    {nameList[3*i]} & {nameList[3*i+1]} & {nameList[3*i+2]} \\\\", file=texfile)
if len(nameList)%3==1:
    print(f"    {nameList[-1]} & & \\\\", file=texfile)
elif len(nameList)%3==2:
    print(f"    {nameList[-2]} & {nameList[-1]} & \\\\", file=texfile)
print("  \end{tabular}", file=texfile)
print("  \end{center}", file=texfile)
print("  \caption{Input features}", file=texfile)
print(f"  \label{{tab:{outDir}features}}", file=texfile)
print("\end{table}", file=texfile)
texfile.close()

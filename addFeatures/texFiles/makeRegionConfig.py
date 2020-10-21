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

inDF = pd.read_csv(inFile, nrows=1e6)
inDF = inDF.dropna()

texfile = open('regions_'+outDir+".config", "w")

icount = 1

print("\n%---------------------------", file=texfile)
print("%         REGIONS           ", file=texfile)
print("%---------------------------\n", file=texfile)

for c in inDF:
    if c=='match': continue
    print(f'"{c}"')

    xName, r = name(c)
    print(f"Region: {c}", file=texfile)
    print(f"  Type: VALIDATION", file=texfile)
    print(f"  Label: \"{c}\"", file=texfile)
    print(f"  TexLabel: \"{xName}\"", file=texfile)
    varTitle = xName.replace('$', '').replace('_T','_{T}').replace('_0', '_{0}').replace('_1', '_{1}').replace('_2', '_{2}')
    print(f"  VariableTitle: \"{varTitle}\"", file=texfile)
    if '[GeV]' in xName:
        print(f"  Variable: \"{c}/1000\",30,{r[0]},{r[1]}\n", file=texfile)
    else:
        print(f"  Variable: \"{c}\",30,{r[0]},{r[1]}\n", file=texfile)

texfile.close()

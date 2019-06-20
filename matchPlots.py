import pandas
import xgboost as xgb
import pickle
import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

inFile = sys.argv[1]
inDF = pd.read_csv(inFile)

good = inDF[inDF['decay']==1]
bad = inDF[inDF['decay']==0]

outDir = sys.argv[2]

for c in inDF:
    if 'MV2c10' in c:
        r = (-1, 1)
    elif 'dR' in c:
        r = (0, 6)
    else:
        r = (0, 300000)

    plt.figure()
    plt.hist(good[c], 30, alpha=0.5, range=r, label="semi leptonic")
    plt.hist(bad[c][:good.shape[0]], 30 ,range = r, alpha=0.5, label="fully leptonic")
    plt.legend()
    plt.xlabel(c)
    plt.savefig(outDir+c+".png")
    plt.close()

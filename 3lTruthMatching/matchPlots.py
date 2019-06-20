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

good = inDF[inDF['match']==1]
bad = inDF[inDF['match']==0]

outDir = sys.argv[2]

for c in inDF:
    if 'MV2c10' in c:
        r = (-1, 1)
    elif 'dR' in c:
        r = (0, 6)
    elif 'numTrk' in c:
        r = (0, 15)
    elif 'jvt' in c:
        r = (0,1)
    else:
        r = (0, 300000)

    plt.figure()
    plt.hist(good[c], 30, alpha=0.5, range=r, label="Correct")
    plt.hist(bad[c][:good.shape[0]], 30 ,range = r, alpha=0.5, label="Incorrect")
    plt.legend()
    plt.xlabel(c)
    plt.savefig(outDir+c+".png")
    plt.close()

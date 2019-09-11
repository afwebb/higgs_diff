import pandas
import xgboost as xgb
import pickle
import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')

inFile = sys.argv[1]
inDF = pd.read_csv(inFile, nrows=100000)

good = inDF[inDF['match']==1]
bad = inDF[inDF['match']==0]

outDir = sys.argv[2]

for c in inDF:
    if 'MV2c10' in c:
        r = (-1, 1)
    elif 'dR' in c:
        r = (0, 6)
    else:
        r = (0, 300000)

    plt.figure()
    plt.hist(good[c], 30, alpha=0.5, range=r, label="Correct Match")
    plt.hist(bad[c][:good.shape[0]], 30 ,range = r, alpha=0.5, label="Bad Match")
    plt.legend()
    plt.xlabel(c)
    plt.savefig(outDir+c+".png")
    plt.close()

f = plt.figure(figsize=(19,15))
plt.matshow(inDF.corr(),fignum=f.number)
for (i,j), z in np.ndenumerate(inDF.corr()):
    plt.text(j,i, '{:0.2f}'.format(z), ha='center', va='center')
plt.xticks(range(inDF.shape[1]), inDF.columns, fontsize=14, rotation=45)
plt.yticks(range(inDF.shape[1]), inDF.columns, fontsize=14)
cb = plt.colorbar()
#cb.ax.tick
#plt.title('Top Match Feature Correlations', fontsize=16)
plt.savefig(outDir+"CorrMat.png")
plt.close()

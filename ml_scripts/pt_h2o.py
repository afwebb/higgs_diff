import pandas as pd
import numpy as np
import re
import sklearn
import h2o
h2o.init()
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid import H2OGridSearch
#import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sys
import scipy
import os
from sklearn.neighbors import KernelDensity

h2o.remove_all()

outDir = sys.argv[1]

'''
#Reading the data
#inDF = pd.read_csv('../inputData/67_4j.csv')
inDF = pd.read_csv('../inputData/'+dsid+'_'+njet+'j.csv')

#change higgs_pt from continuous to categorical
#inDF.loc[ inDF['higgs_pt'] <= 300000, 'higgs_pt'] = 0
#inDF.loc[ inDF['higgs_pt'] > 300000, 'higgs_pt'] = 1 ;

inDF.shape

train, test = train_test_split(inDF, test_size=0.3)
train.columns
train.shape
test.shape

#train = train.drop(['higgs_pt'],axis=1)
#test = test.drop(['higgs_pt'],axis=1)
'''

h2o_train = h2o.import_file(path='/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_train_'+outDir+'.csv', destination_frame='h2o_train')
h2o_test = h2o.import_file(path='/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_test_'+outDir+'.csv', destination_frame='h2o_test')

h2o_model = H2ODeepLearningEstimator(hidden=[250,250,250,250,250,250,250,250,250],
                                     epochs= 5000,
                                     train_samples_per_iteration= 5000,
                                     rate= 0.001,
                                     #l1=1e-5,
                                     #l2=1e-5,
                                     #max_w2=10,
                                     stopping_tolerance=1e-9)
                                     

#nn_grid#.set_params(hidden=[150,150,150,150,150,150,150,150], epochs=1000, train_samples_per_iteration=10000)
h2o_model.train(x=h2o_train.names.remove('higgs_pt'), y = 'higgs_pt',
                #x=h2o_train.names[1:2]+h2o_train.names[4:], y = h2o_train.names[3],
                training_frame = h2o_train,
                validation_frame = h2o_test)

print(h2o_model.rmse(valid=True))

print(h2o_model.hidden)

n = h2o_model.hidden[0]
l = len(h2o_model.hidden)

model_path = h2o.save_model(h2o_model, path = 'h2o_models/'+outDir, force=True)

plt.figure()
plt.rcdefaults()
fig, ax = plt.subplots()
variables = h2o_model._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = h2o_model._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+outDir+'/h2o_var_important.png')

y_train_pred = h2o_model.predict(h2o_train)
y_test_pred = h2o_model.predict(h2o_test)

y_train = h2o_train[3].as_data_frame().values
y_test = h2o_test[3].as_data_frame().values

y_train_pred = y_train_pred.as_data_frame().values
y_test_pred = y_test_pred.as_data_frame().values

plt.figure()
plt.hist(y_test, 20, log=False, range=(0,800000), alpha=0.5, label='truth')
plt.hist(y_test_pred, 20, log=False, range=(0,800000), alpha=0.5, label='test')
plt.title("H2O Test Data, layers=%i, nodes=%i, loss=%0.4f" %(l, n, h2o_model.rmse(valid=True)))
plt.xlabel('Higgs Pt')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/h2o_test_pt_spectrum_'+str(l)+'l_'+str(n)+'n.png')
'''
xy = np.vstack([y_train[:100], y_train_pred[:100]])
#xy = np.where(xy==0, xy, 0.0001)
xy = np.nan_to_num(xy)

print(np.isnan(xy).any())
print(np.isinf(xy).any())
print(xy)
#print(min(xy))
#print(max(xy))
z = scipy.stats.gaussian_kde(xy)(xy)
#kde = KernelDensity(bandwidth=0.04, kernel='gaussian', algorithm='ball_tree')
#kde.fit(xy)

#z = kde.score_samples(xy)
#np.reshape(z, y_test[:100])

plt.figure()
plt.scatter(y_train[:100], y_test[:100], c=np.log(z), edgecolor='')
plt.title("H2O Test Data, loss=%0.4f" %(l, n, h2o_model.rmse(valid=True)))
plt.xlabel('Truth Pt')
plt.ylabel('Predicted Pt')
plt.plot([0,800000],[0,800000], zorder=10)
plt.savefig('plots/'+outDir+'/h2o_test_pt_scatter_'+str(l)+'l_'+str(n)+'n.png')
'''
cutoff = [150000]

plt.figure()
for c in cutoff:
    yTest = np.where(y_test > c, 1, 0)
    ypTest = y_test_pred

    auc = sk.metrics.roc_auc_score(yTest,ypTest)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

    plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

    yTrain = np.where(y_train > c, 1, 0)
    ypTrain = y_train_pred

    auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
    fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)

    plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))

plt.title("H2O ROC")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/h2o_roc_'+str(l)+'l_'+str(n)+'n.png')

#plt.figure()
#perf = best_model.model_performance()
#perf.plot()
#plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+outDir+'/h2o_roc.png')



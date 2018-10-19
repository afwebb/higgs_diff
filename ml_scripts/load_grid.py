import h2o
h2o.init()
import numpy as np

results=[]

for f in open('h2o_models.txt', 'r'):
    m = h2o.load_model(f.rstrip())
    results.append([m.auc(valid=True),m])

best_auc = sorted(results, key=lambda x: x[0])[-1]
best = best_auc[1]
params = ['epochs', 'train_samples_per_iteration','hidden','rate']

print( 'auc: '+str(best_auc[0]))
for p in params:
    print( p+': '+str(best.params[p]['actual']))


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.figure()
plt.rcdefaults()
fig, ax = plt.subplots()
variables = best._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = best._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.savefig('plots/h2o_var_important.png')

plt.figure()
perf = best.model_performance()
perf.plot()
plt.savefig('plots/h2o_roc.png')

#params = best[1].get_params(deep=False)
#print( params['hidden'])
#print( params['epochs'])
#print( params['train_samples_per_iteration'])
#print( best[0])



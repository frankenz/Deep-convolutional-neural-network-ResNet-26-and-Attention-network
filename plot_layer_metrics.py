import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
import json
import glob
import numpy as np

# method I: plt
import matplotlib.pyplot as plt
#plt.title('Receiver Operating Characteristic')
from matplotlib import cm
import sys

path = sys.argv[1]
#path = 'glioblastoma/run_E500_yTYPE_GBM_RESNET/'
path = path + '*summary.json'
cmap_lin = cm.rainbow(np.linspace(0,1,len(glob.glob(path))))
results={} 

with open(glob.glob(path)[0], 'r') as json_file:
    d_trainsplit_load = json.load(json_file)
    for key in d_trainsplit_load['model_max_weights']:
        if 'bias' in key: continue    
        results[key] = []

for i, file in enumerate(sorted(glob.glob(path))):
    with open(file, 'r') as json_file:
        d_trainsplit_load = json.load(json_file)
        for key in d_trainsplit_load['model_max_weights']:
            if 'bias' in key: continue    
            results[key].append(d_trainsplit_load['model_max_weights'][key])

print (results)
color=iter(cm.rainbow(np.linspace(0,1,len(results.keys()))))
plt.figure(figsize=(8,8))
for key in results.keys():
    c=next(color)
    plt.plot(results[key], c=c, label = key)

plt.ylabel('Value')
plt.xlabel('Epoch')
#plt.ylim(0, 2.0)
plt.legend(loc = 'best')
plt.show()
plt.savefig("/home/andrew/Dropbox/DCPL//gbm_layer_progress.pdf")

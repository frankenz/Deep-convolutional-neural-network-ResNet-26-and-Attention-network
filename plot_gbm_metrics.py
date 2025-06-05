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

def plot_gbm_metrics(path):

    #path = 'glioblastoma/run_E500_yTYPE_GBM_RESNET/'
    path = path + '*summary.json'
    cmap_lin = cm.rainbow(np.linspace(0,1,len(glob.glob(path))))
    
    
    tloss = []
    vloss = []
    tacc = []
    vacc = []
    twsum = []
    vwsum = []
    rsum = []
    temp = []
    
    vf1_A = []
    vf1_B = []
    vf1_C = []
    tf1_A = []
    tf1_B = []
    tf1_C = []
    vauc = []
    tauc = []
    
    for i, file in enumerate(sorted(glob.glob(path))):
        with open(file, 'r') as json_file:
               d_trainsplit_load = json.load(json_file)
        vf1_A.append( d_trainsplit_load['valid_acc']['A']['f1-score'] )
        vf1_B.append( d_trainsplit_load['valid_acc']['B']['f1-score'] )
        vf1_C.append( d_trainsplit_load['valid_acc']['C']['f1-score'] )
        tf1_A.append( d_trainsplit_load['train_acc']['A']['f1-score'] )
        tf1_B.append( d_trainsplit_load['train_acc']['B']['f1-score'] )
        tf1_C.append( d_trainsplit_load['train_acc']['C']['f1-score'] )
    
        tacc.append( d_trainsplit_load['train_acc']['accuracy'] )
        vacc.append( d_trainsplit_load['valid_acc']['accuracy'] )
    
        tloss.append( d_trainsplit_load['train_loss'] )
        vloss.append( d_trainsplit_load['valid_loss'] )
        twsum.append( d_trainsplit_load['train_wsum'] )
        vwsum.append( d_trainsplit_load['valid_wsum'] )
        rsum.append( d_trainsplit_load['train_sum'] )
        temp.append( d_trainsplit_load['model_temp'] )
    
    print (tloss)
    plt.figure(figsize=(8,8))
    plt.plot(tloss, 'C1--', label = 'Train Loss')
    plt.plot(vloss, 'C1', label = 'Valid Loss')
    plt.plot(twsum, 'C6--', label = 'Train Regularization')
    plt.plot(vwsum, 'C6', label = 'Valid Regularization')
    #plt.plot(rsum, 'C8', label = 'Average Sum')
    #plt.plot(temp, 'C7', label = 'Model Temperature')
    plt.plot(tacc, 'k--', label = 'Train Accuracy')
    plt.plot(vacc, 'k', label = 'Validation Accuracy')
    plt.plot(vf1_A, 'r', label = 'Validation A F1-Score')
    plt.plot(vf1_B, 'g', label = 'Validation B F1-Score')
    plt.plot(vf1_C, 'b', label = 'Validation C F1-Score')
    plt.plot(tf1_A, 'r--', label = 'Train A F1-Score')
    plt.plot(tf1_B, 'g--', label = 'Train B F1-Score')
    plt.plot(tf1_C, 'b--', label = 'Train C F1-Score')
    
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    #plt.ylim(0, 2.0)
    plt.legend(loc = 'upper left')
    #plt.show()
    plt.savefig("/home/andrew/Dropbox/DCPL//gbm_progress.pdf")

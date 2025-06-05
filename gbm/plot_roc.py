import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
import json
import glob
import numpy as np

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
from matplotlib import cm

def plot_roc(path, tag='TEST', show=False, describe=False):

    path = path + '/*predictions.json'
    cmap_lin = cm.rainbow(np.linspace(0,1,len(glob.glob(path))))
    aucs = []
    for i, file in enumerate(sorted(glob.glob(path))):
        with open(file, 'r') as json_file:
               d_trainsplit_load = json.load(json_file)
        preds = d_trainsplit_load['predictions']
        labels = d_trainsplit_load['labels']
        fpr, tpr, threshold = metrics.roc_curve(labels, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color=cmap_lin[i], label = 'AUC = %0.2f' % roc_auc)
        print (roc_auc, file)
        aucs.append(roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_test.pdf')
    plt.show()
    plt.close()
    plt.plot(aucs)
    plt.savefig('auc_test.pdf')
    plt.show()


if __name__== "__main__":
    path = sys.argv[1]
    print ("Hey!")
    plot_gbm_metrics(path, show=True, describe=True)



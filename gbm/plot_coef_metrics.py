import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
import json
import glob
import numpy as np
import pandas as pd
# method I: plt
import matplotlib.pyplot as plt
#plt.title('Receiver Operating Characteristic')
from matplotlib import cm
import sys

def ema(data, window):
    data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def plot_gbm_metrics(path, tag='TEST', show=False):

    #path = 'glioblastoma/run_E500_yTYPE_GBM_RESNET/'
    path = path + '/*summary.json'
    cmap_lin = cm.rainbow(np.linspace(0,1,len(glob.glob(path))))

    c1 = []
    c2 = []
    c3 = []

    for i, file in enumerate(sorted(glob.glob(path))):
        with open(file, 'r') as json_file:
               d_trainsplit_load = json.load(json_file)
        c1.append( d_trainsplit_load['coef_a1'] )
        c2.append( d_trainsplit_load['coef_a2'] )
        c3.append( d_trainsplit_load['coef_a3'] )

    plt.figure(figsize=(8,8))
    plt.plot(c1, 'r', label = 'Coefficent 1')
    plt.plot(c2, 'g', label = 'Coefficent 2')
    plt.plot(c3, 'b', label = 'Coefficent 3')


    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc = 'upper left')
    plt.savefig(f"/home/andrew/Dropbox/DCPL//gbm_coef_tag{tag}.pdf")
    if show: plt.show()
    plt.close()

def plot_prediction_summary(epoch, output_dir, preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    plt.figure(figsize=(24,4))
    # set width of bar
    barWidth = 0.2


    # set height of bar
    bars1 = preds[:,0]
    bars2 = preds[:,1]
    bars3 = preds[:,2]

    bars1_colors = np.where(labels==0, 'g', 'r')
    bars2_colors = np.where(labels==1, 'g', 'r')
    bars3_colors = np.where(labels==2, 'g', 'r')

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color=bars1_colors, width=barWidth, edgecolor='white', label='Pred A (r/g correct)')
    plt.bar(r2, bars2, color=bars2_colors, width=barWidth, edgecolor='white', label='Pred B (r/g correct)')
    plt.bar(r3, bars3, color=bars3_colors, width=barWidth, edgecolor='white', label='Pred C (r/g correct)')


    # Create legend & Show graphic
    plt.legend()
    plt.savefig(f"/home/andrew/Dropbox/DCPL/gbm_heatmaps/validation/validation_performance-{str(epoch).zfill(3)}.pdf")
    plt.close()


if __name__== "__main__":
    path = sys.argv[1]
    print ("Hey!")
    plot_gbm_metrics(path, show=True)

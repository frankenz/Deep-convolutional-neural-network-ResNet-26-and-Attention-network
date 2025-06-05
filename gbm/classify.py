import argparse
import random
import math
import time

import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import torch
from torch import nn, optim
from torchvision import utils
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
# Local stuff
from RoiBuilder import RoiBuilder
from model import Attention
from GlioblastomaDS import *
from PyTorchHelpers import *
from cnn_layer_visualization import CNNLayerVisualization

writer = SummaryWriter(max_queue=2)
disc_cutoff = 6
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")

parser = argparse.ArgumentParser(description='Attention based classifier for WSI images using partial GAN trained discriminator reduction')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument(
    '--ckpt', default=None, type=str, help='load from previous checkpoints'
)
parser.add_argument(
    '--epoch_start', default=0, type=int, help='Which epoch to start at?'
)
parser.add_argument(
    '--epoch_end', default=40, type=int, help='Which epoch to start at?'
)
parser.add_argument(
    '--no_from_rgb_activate',
    action='store_true',
    help='use activate in from_rgb (original implementation)',
)
parser.add_argument(
    '--transfer',
    action='store_true',
    help='Transfer learning, reset all linear layers',
)
parser.add_argument(
    '--test_only',
    action='store_true',
    help='Exit after test',
)

args = parser.parse_args()
print (args)


if args.test_only:
    output_dir = f'./test_data'
    try:
        os.mkdir(output_dir)
    except:
        pass
else:
    output_dir = f'./run_attention_classifier_glioblastoma_{timestampStr}'
    os.mkdir(output_dir)

# Turn on gradient
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# Create a dataloader for this epoch
def sample_data(dataset, batch_size=64, image_size=4):
    loader_kwargs = {'num_workers': 4, 'pin_memory': False, 'shuffle': True}
    dataset.NewResolution(image_size, batch_size)
    dataset.SetDropout(1.0)
    loader = DataLoader(dataset, batch_size=1, **loader_kwargs)
    return loader

# Create a dataloader for this epoch
def sample_data_test(dataset, batch_size=64, image_size=4):
    loader_kwargs = {'num_workers': 4, 'pin_memory': False, 'shuffle': False}
    dataset.NewResolution(image_size, batch_size)
    dataset.SetDropout(1.0)
    loader = DataLoader(dataset, batch_size=None, **loader_kwargs)
    return loader

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def visualize (args, epoch, step, classifier, discriminator, sample, show=False):
    print ("===> VISUALIZE: Epoch = ", epoch)
    name = sample.getname()
    data, raster, img_data = sample.get_test_set()

    # Best to put it all on at once to avoid sync latency
    master_features = data.cuda().float()
    bag_label = torch.tensor([1]).float().cuda()
    b_size = 0
    with torch.no_grad():
        A, loss, entropy, pred, error = classifier(master_features, bag_label, step_input=disc_cutoff)
        A1 = A[0].cpu()

    create_map(name, epoch, step, img_data, raster, A1, "A1", show)
    #create_map(epoch, step, img_data, raster, A2, "A2")

def create_map(name, epoch, step, img_data, raster, A, level, show=False):
    x_locs = []
    y_locs = []
    att_weights_zscore = (100.0/torch.max(A).float())*(A.data)
    att_weights_real = A.data
    cmap_lin = cm.rainbow(np.linspace(0,1,101))

    fig, ax = plt.subplots(figsize=(15,15), nrows=1, ncols=1)
    for i,roi in enumerate(img_data):
        x_locs.append(raster[i][1])
        y_locs.append(raster[i][0])
        ax.imshow(roi, origin='upper', extent=(raster[i][1], raster[i][1] + 600, raster[i][0], raster[i][0] - 600))
        # Create a Rectangle patch
        rect = patches.Rectangle((raster[i][1],raster[i][0]-600),600,600,linewidth=1,facecolor=cmap_lin[int(att_weights_zscore[i])], alpha=0.4)
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.ylim(min(x_locs),max(x_locs))
    plt.xlim(min(y_locs),max(y_locs))
    plt.title("Epoch = {0}".format(epoch))
    if show:
        plt.show()
    plt.savefig(f'/home/andrew/Dropbox/DCPL/gbm_heatmaps/train_epoch-{str(epoch).zfill(3)}_step-{str(step)}_sample-{name}_attlevel-{level}.pdf')
    plt.close()

def test(args, epoch, dataset, classifier, discriminator, global_steps=0):

    w1 = 1.0
    w2 = 2.0
    print ("===> TEST: Epoch, w1, w2: = ", epoch, w1, w2)

    train_error = 0.

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    dataset.test()
    loader = sample_data_test( dataset, batch_size=128, image_size=128)  # Get a new dataloader with updated parameters


    f_tomove_img = open(f"{output_dir}/move_images.sh","w+")
    f_imagemanifest = open(f"{output_dir}/manifest_image.sh","w+")
    f_headmanifest = open(f"{output_dir}/manifest_heat.csv","w+")

    f_imagemanifest.write("path,studyid,clinicaltrialsubjectid,imageid\n")
    f_headmanifest.write("path,studyid,clinicaltrialsubjectid,imageid\n")
    predictions = []
    labels = []

    for batch_idx, master_batch_data in enumerate(loader):

        metadata = master_batch_data[3]
        if not metadata['caMIC_eligable']: continue

        print (metadata)
        f_imagemanifest.write("{0},{1},{2},{2}\n".format(metadata['camic_id'], metadata['studyid'], metadata['basename'], metadata['basename']))
        f_tomove_img.write("cp '{0}' /home/andrew/install/quip_distro/images/gbm_validation_set/\n".format(metadata['fullpath']))

        raster = master_batch_data[2].data.numpy()
        bag_label = master_batch_data[1].cuda().float()
        master_features = master_batch_data[0].cuda().float()
        b_size = master_features.shape[0]

        with torch.no_grad():
            A, loss, entropy, pred, error = classifier(master_features, bag_label, step_input=disc_cutoff)
            attn, activations = classifier.tile_activation(master_features, bag_label, step_input=disc_cutoff)

        i_error = error.item()
        i_label = bag_label.item()

        predictions.append(pred.cpu().numpy())
        labels.append(i_label)

        train_error  += error.item()
        error_rate   = 100.0 * train_error / (1.0 + batch_idx)
        write_map(metadata, epoch, raster, attn.cpu(), activations.cpu())

    f_tomove_img.close()
    f_imagemanifest.close()
    f_headmanifest.close()
    target_names = ['A', 'B', 'C']
    print(classification_report(labels, predictions, target_names=target_names))

def write_map(meta, epoch, raster, attn, activations):
    name = meta['basename']
    att_weights_real = plt.Normalize()(attn.data)
    f= open(f'{output_dir}/prediction-AGMIL-ATTN.{name}.dla',"w+")
    for i,coord in enumerate(raster):
        f.write(f'{coord[1]} {coord[0]} {att_weights_real[i,0]}\n')
    f.close()
    f= open(f'{output_dir}/prediction-AGMIL-ACTF1.{name}.dla',"w+")
    for i,coord in enumerate(raster):
        f.write(f'{coord[1]} {coord[0]} {activations[i,0]}\n')
    f.close()
    f= open(f'{output_dir}/prediction-AGMIL-ACTF2.{name}.dla',"w+")
    for i,coord in enumerate(raster):
        f.write(f'{coord[1]} {coord[0]} {activations[i,1]}\n')
    f.close()
    f= open(f'{output_dir}/prediction-AGMIL-ACTF3.{name}.dla',"w+")
    for i,coord in enumerate(raster):
        f.write(f'{coord[1]} {coord[0]} {activations[i,2]}\n')
    f.close()

def validate(args, epoch, dataset, classifier, discriminator, global_steps=0):

    w1 = 1.0
    w2 = 2.0
    print ("===> VALIDATION: Epoch, w1, w2: = ", epoch, w1, w2)

    train_error = 0.

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    dataset.eval()
    loader = sample_data( dataset, batch_size=128, image_size=128)  # Get a new dataloader with updated parameters

    pbar = tqdm(range(len(loader)))
    predictions = []
    labels = []

    for batch_idx, master_batch_data in enumerate(loader):

        bag_label = master_batch_data[1].cuda().float().squeeze(0)
        master_features = master_batch_data[0].squeeze(0).cuda().float()
        b_size = master_features.shape[0]

        with torch.no_grad():
            A, loss, entropy, pred, error = classifier(master_features, bag_label, step_input=disc_cutoff)

        i_error = error.item()
        i_label = bag_label.item()

        predictions.append(pred.cpu().numpy())
        labels.append(i_label)

        train_error  += error.item()
        error_rate   = 100.0 * train_error / (1.0 + batch_idx)


        state_msg = (
            f'Tiles: {b_size:d}; Validation Error Rate: {error_rate:.3f} %'
        )
        pbar.set_description(state_msg)
        pbar.update()
    pbar.close()

    target_names = ['A', 'B', 'C']
    print(classification_report(labels, predictions, target_names=target_names))



def train(args, epoch, dataset, classifier, discriminator, global_steps=0):

    w1 = 1.0
    w2 = 2.0
    print ("===> TRAIN: , w1, w2: = ", epoch, w1, w2)

    train_loss = 0.
    train_error = 0.
    train_entropy = 0.
    TOTAL_LOSS = 0
    count = 0
    step = 0

    requires_grad(classifier,    True)
    optimizer.zero_grad()

    dataset.train()
    loader = sample_data( dataset, batch_size=128, image_size=128)  # Get a new dataloader with updated parameters

    pbar = tqdm(range(len(loader)))
    predictions = []
    labels = []

    for batch_idx, master_batch_data in enumerate(loader):

        bag_label = master_batch_data[1].cuda().float().squeeze(0)
        master_features = master_batch_data[0].squeeze(0).cuda().float()

        b_size = master_features.shape[0]

        A, loss, entropy, pred, error = classifier(master_features, bag_label, step_input=disc_cutoff)
        A.detach()

        predictions.append(pred.cpu().numpy())
        labels.append(bag_label.item())

        train_entropy += entropy.item()
        train_loss    += loss.item()
        train_error   += error.item()

        TOTAL_LOSS += w1*loss + w2*entropy

        count += 1
        if count >= 4:
            TOTAL_LOSS.backward()

            #plot_grad_flow(classifier.named_parameters(), writer, global_steps)
            if (global_steps) % 5 == 0: plot_layer_summary(classifier.named_parameters(), epoch, global_steps, output_dir=output_dir)
            plot_attn_flow("Attention Distribution", A, writer, global_steps)
            #plot_bag_flow("Bag-level Activation",    M, writer, global_steps)

            optimizer.step()
            classifier.step()
            optimizer.zero_grad()
            global_steps += 1


            TOTAL_LOSS = 0.0
            step += 1
            count = 0

        entropy_rate = 1.0 * train_entropy / (1.0 + batch_idx)
        loss_rate    = 1.0 * train_loss    / (1.0 + batch_idx)
        error_rate   = 100.0 * train_error / (1.0 + batch_idx)

        state_msg = (
            f'Tiles: {b_size:d}; Batch Train Loss: {loss_rate:.3f}; Batch Sum of Weights: {entropy_rate:.8f}; Epoch Error Rate: {error_rate:.3f} %'
        )
        pbar.set_description(state_msg)
        pbar.update()

    torch.save(
        {
            'classifier':    classifier.state_dict(),
            'optimizer':     optimizer.state_dict(),
        },
        f'{output_dir}/train_step-{str(epoch).zfill(3)}.model',
    )
    pbar.close()
    target_names = ['A', 'B', 'C']
    print(classification_report(labels, predictions, target_names=target_names))

if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1


    classifier = Attention(params={'temp':0.0, 'beta':0.99}).cuda()

    optimizer = torch.optim.Adam(   [
        {"params": classifier.cnn.parameters(),        "lr": 20.0 * args.lr},
        {"params": classifier.attention.parameters(),  "lr": 2.00 * args.lr},
        {"params": classifier.classifier.parameters(), "lr": 1.00 * args.lr},
    ],
    betas=(0.9,0.99), lr=args.lr)

    if args.ckpt is not None:
        print ("Loading checkpoint!!!")
        ckpt = torch.load(args.ckpt)
        classifier.load_state_dict(ckpt['classifier'])
#        optimizer.load_state_dict(ckpt['optimizer'])

    if args.transfer:
        print ("Randomizing Linear Layers!!!")
        classifier.reset_linear()

    dataset = GHPSingleBagDatasetSimple(bag=True, output_dir=output_dir)
    dataset.load_from_checkpoint('./training_validation_testing_data11-Nov-2019-18-44-05.json')

    global_steps = 0

    test_roi1 = RoiBuilder('/raid/GHP Immunohistochemistry/H&E/GHP_269_B5_H&E.scn') #B
    test_roi1.update_resolution_and_buffer(128, 128)

    test_roi2 = RoiBuilder('/raid/GHP Immunohistochemistry/H&E/GHP_257_B2_H&E.scn') #A
    test_roi2.update_resolution_and_buffer(128, 128)

    test_roi3 = RoiBuilder('/raid/GHP Immunohistochemistry/H&E/GHP_215_C1_H&E.scn') # C
    test_roi3.update_resolution_and_buffer(128, 128)

    if args.test_only:
        dataset_test = GHPSingleBagDatasetSimple(bag=True, output_dir=output_dir)
        dataset_test.load_new()
        test (args, args.epoch_start - 1, dataset_test, classifier, None, global_steps)
        exit()

    for ep in range(args.epoch_start, args.epoch_end):
        train   (args, ep, dataset, classifier, None, global_steps)
        validate(args, ep, dataset, classifier, None, global_steps)
        visualize (args, ep, "Last",  classifier, None, sample=test_roi1)
        # visualize (args, ep, "Last",  classifier, None, sample=test_roi2)
        # visualize (args, ep, "Last",  classifier, None, sample=test_roi3)

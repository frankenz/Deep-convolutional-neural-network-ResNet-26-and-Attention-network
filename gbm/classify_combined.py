import argparse
import random
import math
import time

import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import torch
from torch import nn, optim
from torchvision import utils
from torchsummary import summary
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# SKLearn stuff
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from scipy import stats

# Local stuff
from RoiBuilder import RoiBuilder
from GlioblastomaDS import *
from PyTorchHelpers import *
from model import Attention

from plot_gbm_metrics import plot_gbm_metrics, plot_prediction_summary

# ===================================================================================================================================================

disc_cutoff = 6
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")

# ===================================================================================================================================================

parser = argparse.ArgumentParser(description='Attention based classifier for WSI images using partial GAN trained discriminator reduction')

parser.add_argument(
    '--tag', default="TEST", type=str, help='Output tag'
)
parser.add_argument(
    '--ckpt', default=None, type=str, help='load from previous checkpoints'
)
parser.add_argument(
    '--epoch_start', default=0, type=int, help='Which epoch to start at?'
)
parser.add_argument(
    '--fold', default=0, type=int, help='Which fold?'
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
    '--peak',
    action='store_true',
    help='Look at weight matrix',
)
parser.add_argument(
    '--test_only',
    action='store_true',
    help='Exit after test',
)
parser.add_argument(
    '--interface',
    action='store_true',
    help='Run in interface mode',
)

args = parser.parse_args()
print (args)

if args.interface:
    output_dir = f'./interface_data'
    try:
        os.mkdir(output_dir)
    except:
        pass
else:
    output_dir = "run_{0}".format(args.tag)
    try:
        os.mkdir(output_dir)
    except: pass
    try:
        os.mkdir(f"/home/zf2263/Dropbox/gbm_heatmaps/{output_dir}")
    except: pass

#writer = SummaryWriter(log_dir="runs/TAG_{0}".format(args.tag), flush_secs=30)


# ===================================================================================================================================================

def SetStage(optimizer, model, epoch, test=False):
    base_lr = 0.0002
    schedule = [0,10,150,250,340]

    if epoch >= schedule[0] and epoch < schedule[1]:
        new_lr = base_lr / (schedule[1] - epoch)
        for param_group in optimizer.param_groups: param_group['lr'] = new_lr 
        model.train()
        print ("Stage = [Warmup], lr = [{0}], model mode = [train]".format(new_lr))
    if epoch >= schedule[1] and epoch < schedule[2]:
        for param_group in optimizer.param_groups: param_group['lr'] = base_lr
        model.train()
        print ("Stage = [Main], lr = [{0}], model mode = [train]".format(base_lr))
    if epoch >= schedule[2] and epoch < schedule[3]:
        new_lr = base_lr / 2.0
        for param_group in optimizer.param_groups: param_group['lr'] = new_lr 
        if test: model.eval()
        else:    model.train()
        print ("Stage = [Check], lr = [{0}], training mode = [{1}]".format(new_lr, model.training))
    if epoch >= schedule[3] and epoch < schedule[4]:
        new_lr = base_lr / 10.0
        for param_group in optimizer.param_groups: param_group['lr'] = new_lr 
        if test: model.eval()
        else:    model.train()
        print ("Stage = [Freeze], lr = [{0}], training mode = [{1}]".format(new_lr, model.training))
    if epoch > schedule[4]:
        print ("Stage = [Stop], lr = [{0}], model mode = [save]".format(0.0))
        torch.save({'classifier':    classifier.state_dict(),}, f'{output_dir}/train_step-{str(epoch).zfill(3)}_FINAL.model',)
        exit()

# ===================================================================================================================================================

def visualize (args, epoch, step, classifier, discriminator, sample, show=False, mode="Train"):
    print ("===> VISUALIZE: Epoch = ", epoch)
    name = mode + "-" + sample.getname()
    data, raster, img_data = sample.get_inference_data()
    print (data.shape)
    master_features = data.cuda().float()
    bag_label = torch.tensor([1]).float().cuda()
    classifier.eval()
    with torch.no_grad():
        output = classifier(master_features, bag_label)
        A = output['wROIs'].cpu()
        B = output['Bterm'].cpu()
        M = output['Mterm'].cpu()
        F = output['Fterm'].cpu()

        angles = []
        for m_i, v1 in enumerate(M):
            for m_j, v2 in enumerate(M):
                if m_j > m_i : angles.append(np.arccos(v1.dot(v2) / (v1.norm()*v2.norm()+1e-5)).item())
        angle = np.degrees(np.mean(angles))

        A1 = ((A - A.min())/(A.max() - A.min())).data
        B1 = F.view(F.shape[0], 8, 10).data
        M1 = M.view(3, 1, 1).permute(1,2,0).abs().data # view (chan, H, W)

    create_map (name, epoch, step, img_data, raster, A1, B1, M1, show=False, angle=angle)

def create_map(name, epoch, step, img_data, raster, A, B, M, show=False, angle=0):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(12,8), nrows=2, ncols=3)
    cmap_lin = cm.jet(np.linspace(0,1,105))
    fig.suptitle("Epoch = {0}".format(epoch))

    x_locs = []
    y_locs = []

    A_ALL = (1/3) * (A[0] + A[1] + A[2])

    att_weights_norm_0 = 100 * A_ALL #(100.0/torch.max(A_ALL).float())*(A_ALL)
    att_weights_norm_1 = 100 * A[0]  #(100.0/torch.max(A[0]).float())*(A[0])
    att_weights_norm_2 = 100 * A[1]  #(100.0/torch.max(A[1]).float())*(A[1])
    att_weights_norm_3 = 100 * A[2]  #(100.0/torch.max(A[2]).float())*(A[2])

    tissue_plots = [ax[0,0], ax[0,1], ax[1,0], ax[1,1] ,ax[1,2]]
    for i,roi in enumerate(img_data):
        x_locs.append(raster[i][1])
        y_locs.append(raster[i][0])
        ax[0,0].imshow(roi, origin='upper', extent=(raster[i][1], raster[i][1] + 1200, raster[i][0], raster[i][0] - 1200))
        # Create a Rectangle patch
        if att_weights_norm_0[i] > 0.0:
            rect = patches.Rectangle((raster[i][1],raster[i][0]-1200),1200,1200,linewidth=1,facecolor=cmap_lin[int(att_weights_norm_0[i])], alpha=0.3)
            ax[0,0].add_patch(rect)
        if att_weights_norm_1[i] > 0.0:
            rect = patches.Rectangle((raster[i][1],raster[i][0]-1200),1200,1200,linewidth=1,facecolor=cmap_lin[int(att_weights_norm_1[i])], alpha=0.9)
            ax[1,0].add_patch(rect)
        if att_weights_norm_2[i] > 0.0:
            rect = patches.Rectangle((raster[i][1],raster[i][0]-1200),1200,1200,linewidth=1,facecolor=cmap_lin[int(att_weights_norm_2[i])], alpha=0.9)
            ax[1,1].add_patch(rect)
        if att_weights_norm_3[i] > 0.0:
            rect = patches.Rectangle((raster[i][1],raster[i][0]-1200),1200,1200,linewidth=1,facecolor=cmap_lin[int(att_weights_norm_3[i])], alpha=0.9)
            ax[1,2].add_patch(rect)
        ax[0,1].imshow(B[i], origin='upper', extent=(raster[i][1] + 16, raster[i][1] + 1200 - 16, raster[i][0] - 16, raster[i][0] - 1200 + 16))

    chan_max  = M.max()
    chan_min  = M.min()
    cs = ax[0,2].imshow((M - chan_min)/(chan_max - chan_min), origin='upper', extent=(0, 1, 0, -1) )
    ax[0,2].title.set_text('Angle = {0:.2f}, Chan = {1:.2f},{2:.2f}'.format(angle, float(chan_min), float(chan_max)))
    for a in tissue_plots:
        a.set_ylim(0-1200,np.max(y_locs))
        a.set_xlim(0,np.max(x_locs)+1200)
        a.set_aspect('equal')

    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(f'/home/zf2263/Dropbox/gbm_heatmaps/{output_dir}/gbm_status-{str(epoch).zfill(3)}_sample-{name}-heatmap.pdf')
    plt.close()

# ===================================================================================================================================================
def interface(args, epoch, dataset, classifier, discriminator):
    print ("===> INTERFACING TO CAMICROSCOPE")

    classifier.eval()
    dataset.interface()
    loader = sample_data(dataset, image_size=300, shuffle=False)

    f_tomove_img = open(f"{output_dir}/move_images.sh","w+")
    f_imagemanifest = open(f"{output_dir}/manifest_img.csv","w+")
    f_heatmanifest = open(f"{output_dir}/manifest_heat.csv","w+")

    f_imagemanifest.write("path,studyid,clinicaltrialsubjectid,imageid\n")
    f_heatmanifest.write("path,studyid,clinicaltrialsubjectid,imageid\n")
    predictions = []
    labels = []
    ccls = {}
    slide_ebs = {}
    l_ntiles = []
    print (len(loader))
    
    valid_ids = [
        'GHP_43_B1',
        'GHP_265_B2',
        'GHP_191_B1',
        'GHP_307_A1_CE1',
        'GHP_227_B1_CE',
        'GHP_67_A1',
        'GHP_324_B1_CE1',
        'GHP_116_D1',
        'GHP_59_C1',
        'GHP_131_B3',
        'GHP_145_CE3',
        'GHP_318_CE3',
        'GHP_6_B2_C1',
        'GHP_72_B2',
    ]
    for batch_idx, master_batch_data in enumerate(loader):
        print ("==============================================================")

        metadata = master_batch_data[3]
        for key in metadata.keys(): metadata[key] = metadata[key][0]
    #    if not metadata['caMIC_id_name'] in valid_ids: continue
        print (metadata)
        l_ntiles.append(metadata['ntiles'])
        f_imagemanifest.write("{0},{1},{2},{3}\n".format(metadata['caMIC_image_name'], metadata['caMIC_study'], metadata['caMIC_id_name'], metadata['caMIC_id_name']))
        f_tomove_img.write("cp '{0}' /home/zf2263/install/quip_distro/images/gbm_validation_set/\n".format(metadata['fullpath']))

        master_features = master_batch_data[0].squeeze(0).cuda().float()
        bag_label       = master_batch_data[1].squeeze(0).cuda().float()
        raster          = master_batch_data[2].squeeze(0).numpy()
        b_size          = master_features.shape[0]


        with torch.no_grad():
            output = classifier(master_features, bag_label)

        i_error = output['error'].item()
        i_label = bag_label.item()
        ccls     [metadata['Sample Name']] = np.append(output['y_pred'].flatten().cpu().detach().numpy() , output['Aterm_var'].cpu().item())
        slide_ebs[metadata['Sample Name']] = np.append(bag_label.cpu().item(), output['Mterm'].flatten().cpu().detach().numpy())

        predictions.append(output['y_pred_hat'].cpu().numpy())
        labels.append(i_label)
        print (metadata['caMIC_id_name'])
        print ("True label: ", metadata['outcome_item'])
        print ("Activation variance: ", output['Aterm_var'].cpu().numpy())
        print ("Probabilities: ",       output['y_pred'].cpu().numpy())

        write_map(metadata, epoch, raster, output['Aterm'].cpu(), f_heatmanifest, output_dir)

    pd.DataFrame.from_dict(ccls,      orient='index').to_csv('GBMresult_probs_class.csv')
    pd.DataFrame.from_dict(slide_ebs, orient='index').to_csv('GBMdata_slideEBs_class.csv')
    print ("NTILES = ", l_ntiles)
    f_tomove_img.close()
    f_imagemanifest.close()
    f_heatmanifest.close()
    target_names = ['A', 'B', 'C']
    print(classification_report(labels, predictions, target_names=target_names))

# ===================================================================================================================================================

def validate(args, epoch, dataset, classifier, discriminator, epoch_stats={}):
    print ("===> VALIDATION: Epoch = ", epoch)
    SetStage(optimizer, classifier, epoch, test=True)
    valid_error = 0.
    valid_loss = 0
    valid_reg = 0
    valid_reg_mean = 0
    valid_reg_kld = 0
    optimizer.zero_grad()
    dataset.eval()

    loader = sample_data(dataset , image_size=300, shuffle=False)  # Get a new dataloader with updated parameters

    pbar = tqdm(range(len(loader)))
    predictions = []
    labels = []
    predvals = []

    for batch_idx, master_batch_data in enumerate(loader):

        master_features = master_batch_data[0].squeeze(0).cuda().float()
        bag_label       = master_batch_data[1].squeeze(0).cuda().float()
        b_size          = master_features.shape[0]

        output = classifier(master_features, bag_label)

        valid_reg_mean += output['Aterm_mu'].item()
        valid_reg_kld  += output['KLD'].item()
        valid_reg_l2   = output['l2'].item()

        predictions.append(output['y_pred_hat'].cpu().detach().numpy())
        predvals.append(output['y_pred'].flatten().cpu().detach().numpy())
        labels.append(bag_label.item())

        valid_loss    += output['loss'].item()
        valid_error  += output['error'].item()

        valid_reg_mean_rate = valid_reg_mean / (1.0 + batch_idx)
        valid_reg_kld_rate  = valid_reg_kld  / (1.0 + batch_idx)
        loss_rate    = valid_loss  / (1.0 + batch_idx)
        error_rate   = valid_error / (1.0 + batch_idx)
        reg_rate     = valid_reg   / (1.0 + batch_idx)

        state_msg = (
            f'V: Tiles: {b_size:d}; Loss: {loss_rate:.3f}; (AM, KLD, L2): ({valid_reg_mean_rate:.3f}, {valid_reg_kld_rate:.3f}, {valid_reg_l2:.3f}), Error Rate: {100*error_rate:.2f} %'
        )
        pbar.set_description(state_msg)
        pbar.update()
    pbar.close()
    plot_prediction_summary(epoch, output_dir, predvals, labels)
    target_names = ['A', 'B', 'C']
    epoch_stats['valid_acc'] = classification_report(labels, predictions, target_names=target_names, output_dict=True)
    epoch_stats['valid_loss']  = loss_rate
    epoch_stats['valid_err']   = error_rate
    epoch_stats['valid_wsum']   = valid_reg_mean_rate
    epoch_stats['valid_kld']   = valid_reg_kld_rate



# ===================================================================================================================================================

def peak(args, epoch, dataset, classifier, discriminator, epoch_stats={}):
    print ("===> PEAK: Epoch = ", epoch)

    dataset.train()
    loader = sample_data(dataset , image_size=300)  # Get a new dataloader with updated parameters

    my_act = prime_activation_summary(classifier)
    my_red = prime_activation_vis(classifier)

    for batch_idx, master_batch_data in enumerate(loader):
        master_features = master_batch_data[0].squeeze(0).cuda().float()
        bag_label       = master_batch_data[1].squeeze(0).cuda().float()
        b_size          = master_features.shape[0]

        # Our foward pass
        with torch.no_grad():
            output = classifier(master_features, bag_label)

        for x in my_red.keys(): plot_activations(my_red[x])




# ===================================================================================================================================================

def train(args, epoch, dataset, classifier, discriminator, epoch_stats={}):
    print ("===> TRAIN: Epoch = ", epoch)
    SetStage(optimizer, classifier, epoch)

    epoch_stats['coef_a1']  = torch.sigmoid(10.0*classifier.weight_mask)[0].item()
    epoch_stats['coef_a2']  = torch.sigmoid(10.0*classifier.weight_mask)[1].item()
    epoch_stats['coef_a3']  = torch.sigmoid(10.0*classifier.weight_mask)[2].item()

    step_counter = 0
    train_loss = 0.
    train_error = 0.
    train_reg_mean = 0.
    train_reg_var = 0.
    train_reg_l2 = 0.
    train_reg_kld = 0.
    BATCH_COUNT = 0
    train_sum = 0.

    optimizer.zero_grad()
    dataset.train()


    print ("Current attention coefficients: ", epoch_stats)

    loader = sample_data(dataset , image_size=300)  # Get a new dataloader with updated parameters

    pbar = tqdm(range(len(loader)))
    predictions = []
    labels = []

    my_act = prime_activation_summary(classifier)
    #print("ACT  ",my_act)
    #exit()
    # my_red = prime_activation_vis(classifier)
    for batch_idx, master_batch_data in enumerate(loader):
        master_features = master_batch_data[0].squeeze(0).cuda()
        bag_label       = master_batch_data[1].squeeze(0).cuda()
        b_size          = master_features.shape[0]
        #print ("DATA  ", master_features, bag_label,bag_label.item(), b_size)
        #for input_data in torch.split(master_batch_data, 256):
            #input_data.float()
            #print(input_data.shape)

        # Our foward pass
        output = classifier(master_features, bag_label)
        #print ("OUT  ", output)
        #exit ()
        predictions.append(output['y_pred_hat'].cpu().numpy())
        labels.append(bag_label.item())

        train_reg_mean += output['Aterm_mu'].item()
        train_reg_var  += output['Aterm_var'].item()
        train_reg_l2    = output['l2'].item()
        train_reg_kld  += output['KLD'].item()

        train_loss    += output['loss'].item()
        train_error   += output['error'].item()

        TOTAL_LOSS = output['loss'] 
        TOTAL_LOSS.backward()
        BATCH_COUNT += 1

        if BATCH_COUNT >= 5:
            optimizer.step()
            optimizer.zero_grad()
            step_counter += 1
            BATCH_COUNT = 0

        reg_mean_rate = train_reg_mean / (1.0 + batch_idx)
        reg_var_rate = train_reg_var   / (1.0 + batch_idx)
        reg_kld_rate = train_reg_kld   / (1.0 + batch_idx)
        loss_rate    = train_loss      / (1.0 + batch_idx)
        error_rate   = train_error     / (1.0 + batch_idx)

        state_msg = (
                f'T: Tiles: {b_size:d}; Loss: {loss_rate:.3f}; (AM, AV, KLD, L2): ({reg_mean_rate:.3f}, {reg_var_rate:.3f}, {reg_kld_rate:.3f}, {train_reg_l2:.3f}), Error Rate: {100*error_rate:.2f} %'
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
    epoch_stats['train_acc'] = classification_report(labels, predictions, target_names=target_names, output_dict=True)
    epoch_stats['train_loss'] = loss_rate
    epoch_stats['train_wsum'] = reg_mean_rate
    epoch_stats['train_wvar'] = reg_var_rate
    epoch_stats['train_cll2'] = train_reg_l2
    epoch_stats['train_kld'] = reg_kld_rate
    epoch_stats['train_err']  = error_rate
    epoch_stats['model_mean_weights'] = get_layer_weight_summary_mean(classifier.named_parameters())
    epoch_stats['model_max_weights']  = get_layer_weight_summary_max (classifier.named_parameters())

# ===================================================================================================================================================

if __name__ == '__main__':

    # #
    # # # Skip this unless first time
    # # # dataset.build()
    # # # loader = sample_data( dataset, image_size=300)  # Get a new dataloader with updated parameters
    # # # for batch_idx, master_batch_data in enumerate(loader): print (master_batch_data[0].shape)
    # #
    # dataset.train()

    dataset         = GHPSingleBagDatasetSimple(bag=True, output_dir=output_dir)
    dataset.load_new(n_folds=6, n_fold_selection=args.fold)
    test_roiA   = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/GHP_258_E1_H&E.scn', {}) #A
    train_roiA  = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/GHP_257_B2_H&E.scn', {}) #B
    test_roiB   = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/GHP_317_G1_NE1_H&E.scn', {}) #B
    train_roiB  = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/GHP_251_C2_H&E.scn', {}) #A
    test_roiC   = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/GHP_120_D2_H&E.scn', {}) # C
    train_roiC  = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/GHP_170_D1_H&E.scn', {}) # C
    train_roiSN = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/1012492.svs', {}) # C
    test_roiSN  = RoiBuilder('/raid/GHP Immunohistochemistry/All_HE_scans_GBM_AN/1012458.svs', {}) # C
    test_roiC  .update_resolution_and_buffer(300)
    train_roiC .update_resolution_and_buffer(300)
    test_roiB  .update_resolution_and_buffer(300)
    train_roiB .update_resolution_and_buffer(300)
    test_roiA  .update_resolution_and_buffer(300)
    train_roiA .update_resolution_and_buffer(300)
    train_roiSN.update_resolution_and_buffer(300)
    test_roiSN .update_resolution_and_buffer(300)

    classifier      = Attention(n_classes=3, class_weights=dataset.GetClassWeights()).cuda()
    optimizer       = torch.optim.Adam(classifier.parameters(), betas=(0.9,0.999), lr=0.0002)

    if args.ckpt is not None and not args.transfer:
        print ("Loading Full checkpoint!!!")
        ckpt = torch.load(args.ckpt)
        classifier.load_state_dict(ckpt['classifier'], strict=False)

    if args.ckpt is not None and args.transfer:
        print ("Only transfering Resnet")

        ckpt = torch.load(args.ckpt)['classifier']
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in ckpt.items() if 'cnn' in k and 'conv' in k}
        # 2. overwrite entries in the existing state dict
        ckpt.update(pretrained_dict)
        # 3. load the new state dict
        classifier.load_state_dict(pretrained_dict, strict=False)

    if args.peak:
        print     (classifier.context.inject.weight.cpu().detach().numpy().shape)
        plt.imshow(classifier.context.inject.weight.cpu().detach().numpy())
        plt.show()
        plot_kernels(classifier.named_parameters(), args.epoch_start, 0, output_dir=output_dir)
        # plot_affine(classifier.named_parameters(), args.epoch_start, 0, output_dir=output_dir)
        peak   (args, 0, dataset, classifier, None, {})
        exit()

    if args.epoch_start==0:
            # model_summary(classifier, input_size=(3,300,300), batch_size=2000)
            with open(f'{output_dir}/model_structure.txt', 'w+') as the_file:
                the_file.write(str(classifier))
            # visualize (args, "-1", "Init",  classifier, None, sample=test_roiA,  mode="A_Test")

    if args.interface:
        visualize (args, 0, "Last",  classifier, None, sample=train_roiA, mode="A_Train")
        visualize (args, 0, "Last",  classifier, None, sample=test_roiB,  mode="B_Test")
        visualize (args, 0, "Last",  classifier, None, sample=train_roiB, mode="B_Train")
        visualize (args, 0, "Last",  classifier, None, sample=test_roiC,  mode="C_Test")
        visualize (args, 0, "Last",  classifier, None, sample=train_roiC, mode="C_Train")
        visualize (args, 0, "Last",  classifier, None, sample=test_roiSN,  mode="SN_Test")
        visualize (args, 0, "Last",  classifier, None, sample=train_roiSN, mode="SN_Train")
        interface (args, 0, dataset, classifier, None)

        exit()

    visualize (args, 0, "Last",  classifier, None, sample=test_roiA,  mode="A_Test")
    for ep in range(args.epoch_start, args.epoch_end + 1):
        epoch_stats = {}
        train   (args, ep, dataset, classifier, None, epoch_stats)
        if (ep % 5) == 0:
            validate(args, ep, dataset, classifier, None, epoch_stats)
            savestats (args, output_dir, ep, epoch_stats)
            plot_gbm_metrics(output_dir, args.tag)
 
        if (ep % 10) == 0:
            visualize (args, ep, "Last",  classifier, None, sample=test_roiA,  mode="A_Test")
            visualize (args, ep, "Last",  classifier, None, sample=train_roiA, mode="A_Train")
            visualize (args, ep, "Last",  classifier, None, sample=test_roiB,  mode="B_Test")
            visualize (args, ep, "Last",  classifier, None, sample=train_roiB, mode="B_Train")
            visualize (args, ep, "Last",  classifier, None, sample=test_roiC,  mode="C_Test")
            visualize (args, ep, "Last",  classifier, None, sample=train_roiC, mode="C_Train")
            visualize (args, ep, "Last",  classifier, None, sample=test_roiSN,  mode="SN_Test")
            visualize (args, ep, "Last",  classifier, None, sample=train_roiSN, mode="SN_Train")

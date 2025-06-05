import torch
from torch.utils import data
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import glob
import cv2
import os
import random
import xlrd
import time
import json
from random import randrange

import tifffile
import openslide

from PIL import Image
from PIL import ImageStat

from ctypes import ArgumentError
from RoiBuilder import RoiBuilder
from DataAccessors import split_GHP_convention, getClusterIndex
from sklearn.utils import class_weight

from sklearn.model_selection import KFold




class GHPSingleBagDatasetSimple(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, bag=True, output_dir='.', split=0.90):
        print ("Hello @.@")
        self.train_slide_builders = []
        self.train_slide_outcomes = []
        self.train_slide_path = []
        self.train_slide_record = []
        self.test_slide_builders = []
        self.test_slide_outcomes = []
        self.test_slide_path = []
        self.test_slide_record = []
        self.all_builders = []
        self.all_slide_outcomes = []
        self.all_slide_path = []
        self.all_slide_record = []

        self.ROOT_DIR = '/raid/GHP Immunohistochemistry/'
        self.PATH_KEYS = 'annotations/keys'
        self.PATH_IMG =  'All_HE_scans_GBM_AN'
        self.bagmode = bag
        self.ylabel = "Actual Cluster Designation"
        self.split = split
        self.output_dir = output_dir
        self.studyid = 'gmb-id-nn'

    def load_from_checkpoint(self, SPLIT_DATA_PATH):


        nPCR = 0
        nPCR_test = 0
        d_trainsplit_load = {}
        with open(SPLIT_DATA_PATH, 'r') as json_file:
           d_trainsplit_load = json.load(json_file)

        train_path, train_out = d_trainsplit_load['train_paths'], d_trainsplit_load['train_outcomes']
        validation_path, validation_out = d_trainsplit_load['validation_paths'], d_trainsplit_load['validation_outcomes']

        for file, out in zip(train_path, train_out):
            params = {
                'caMIC_eligable': -1,
                'camic_id': '000',
                'studyid':  "checkpoint-training",
                'pxname':   os.path.split(file)[1].split(' ')[0].split('-')[0],
                'outcome_item': out,
                'outcome_tensor': torch.tensor([out]),
            }
            builder = RoiBuilder(file, params=params)
            self.train_slide_builders.append(builder)
            self.train_slide_outcomes.append(torch.tensor([out]))
            self.train_slide_path.append(file)
            self.train_slide_record.append(out)
            nPCR += out
        for file, out in zip(validation_path, validation_out):
            params = {
                'caMIC_eligable': -1,
                'camic_id': '000',
                'studyid':  "checkpoint-validation",
                'pxname':   os.path.split(file)[1].split(' ')[0].split('-')[0],
                'outcome_item': out,
                'outcome_tensor': torch.tensor([out]),
            }
            builder = RoiBuilder(file, params=params)
            self.test_slide_builders.append(builder)
            self.test_slide_outcomes.append(torch.tensor([out]))
            self.test_slide_path.append(file)
            self.test_slide_record.append(out)
            nPCR_test += out
        print (" >>>>>>>>>>>>>> TRAIN: Done building {0} slides, of which {1} are labeled 1".format(len(self.train_slide_outcomes), nPCR))
        print (" >>>>>>>>>>>>>> VALID: Done building {0} slides, of which {1} are labeled 1".format(len(self.test_slide_outcomes), nPCR_test))

        d_trainsplit = {
            'y-label':              self.ylabel,
            'train_paths':          self.train_slide_path,
            'train_outcomes':       self.train_slide_record,
            'validation_paths':     self.test_slide_path,
            'validation_outcomes':  self.test_slide_record,
        }

        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
        with open('{1}/training_validation_testing_data{0}.json'.format(timestampStr, self.output_dir), 'w') as json_file:
            json.dump(d_trainsplit, json_file, indent=4, sort_keys=True)

    def GetClassWeights(self):
        return torch.tensor(class_weight.compute_class_weight('balanced', np.unique(self.train_slide_record), self.train_slide_record)).float()


    def load_new(self, n_folds=6, n_fold_selection=0):

        nPCR = 0
        nPCR_test = 0
        # Loop through all SCN images (origional, GBM implicated tissue)
        print (os.path.join(self.ROOT_DIR, self.PATH_IMG, '*H&E.scn'))
        print (len(glob.glob(os.path.join(self.ROOT_DIR, self.PATH_IMG, '*H&E.scn'))))
        for file in glob.glob(os.path.join(self.ROOT_DIR, self.PATH_IMG, '*H&E.scn')):

            base_idx = os.path.split(file)[1].split(' ')[0].split('-')[0]
            print ('File: ' + file)
            print ('    Looking for: ' + base_idx)

            # Get the label/outcomne
            identifiers  = split_GHP_convention(base_idx, dtypes=[str,int,str])
            print ('    study: ' + str(identifiers[0]))
            print ('    pxID:  ' + str(identifiers[1]))
            print ('    txID:  ' + str(identifiers[2]))


            outcome=-9
            clus_ind = getClusterIndex(identifiers, self.ylabel, filename="/raid/GHP Immunohistochemistry/PCA Clustering designation and thresholds.xlsx", dtypes=[str,int,str])
            print ('    clus_ind:  ' + str(clus_ind))

            if "Cluster" in self.ylabel:
                if clus_ind=='A': outcome=0
                if clus_ind=='B': outcome=1
                if clus_ind=='C': outcome=2


            print ('    outcome:  ' + str(outcome))

            # Only add ROI if desired label
            if not (outcome==2 or outcome==1 or outcome==0): continue

            params = {
                'caMIC_eligable': -1,   # Will turn if can open in openslide
                'caMIC_image_name': base_idx.replace("_H&E","_HandE"), # The actual image name
                'caMIC_base_name' : base_idx.replace("_H&E","_HandE").split('.')[0], # Image name without extention
                'caMIC_id_name'   : base_idx.replace("_H&E","").split('.')[0], # Image name without extention
                'caMIC_study':    "gbm-classif-nn",
                'caMIC_pxname':   "{0}_{1}_{2}".format(identifiers[0], identifiers[1], identifiers[2]),
                'outcome_item': outcome,
                'outcome_tensor': torch.tensor([outcome]),
            }

            builder = RoiBuilder(file, params=params)
            if builder.getsize() < 20: continue
            self.all_builders.append(builder)
            self.all_slide_outcomes.append(torch.tensor([outcome]))
            self.all_slide_path.append(file)
            self.all_slide_record.append(outcome)

        # Loop through all SVS images (Super normals from cranios)
        for i, file in enumerate(glob.glob(os.path.join(self.ROOT_DIR, self.PATH_IMG, '*.svs'))):
            base_idx = os.path.split(file)[1].split(' ')[0].split('-')[0]

            outcome=1 # All normals

            params = {
                'caMIC_eligable': -1,   # Will turn if can open in openslide
                'caMIC_image_name': base_idx.replace("_H&E","_HandE"), # The actual image name
                'caMIC_base_name' : base_idx.replace("_H&E","_HandE").split('.')[0], # Image name without extention
                'caMIC_id_name'   : base_idx.replace("_H&E","").split('.')[0], # Image name without extention
                'caMIC_study':  "gbm-classif-nn",
                'caMIC_pxname':   "SN_{0}".format(base_idx.replace("_H&E","_HandE").split('.')[0]),
                'outcome_item': outcome,
                'outcome_tensor': torch.tensor([outcome]),
            }

            builder = RoiBuilder(file, params=params)
            if builder.getsize() < 20: continue
            self.all_builders.append(builder)
            self.all_slide_outcomes.append(torch.tensor([outcome]))
            self.all_slide_path.append(file)
            self.all_slide_record.append(outcome)

        folds = list(KFold(n_folds, shuffle=True).split(self.all_builders))
        #folds = list(KFold(n_folds, shuffle=True, random_state=42).split(self.all_builders))
        if n_fold_selection < n_folds:
            n_split = n_folds - 2
            n_select_split = randrange(n_split)+1
            train, test = folds[n_select_split]
            #train, test = folds[n_fold_selection]
        else: 
            print ("Train full dataset")
            train, test = range(len(self.all_builders)), range(len(self.all_builders))
        print (train)
        print (test)
        for idx in train:
            self.train_slide_builders.append    (self.all_builders[idx])
            self.train_slide_outcomes.append    (self.all_slide_outcomes[idx])
            self.train_slide_path.append        (self.all_slide_path[idx])
            self.train_slide_record.append      (self.all_slide_record[idx])
        for idx in test:
            self.test_slide_builders.append     (self.all_builders[idx])
            self.test_slide_outcomes.append     (self.all_slide_outcomes[idx])
            self.test_slide_path.append         (self.all_slide_path[idx])
            self.test_slide_record.append       (self.all_slide_record[idx])

        print (self.train_slide_record)
        print (self.test_slide_record)
        print (" >>>>>>>>>>>>>> TRAIN: Done building {0} slides, of which {1} are labeled 1".format(len(self.train_slide_outcomes), nPCR))
        print (" >>>>>>>>>>>>>> VALID: Done building {0} slides, of which {1} are labeled 1".format(len(self.test_slide_outcomes), nPCR_test))

        d_trainsplit = {
            'train_paths':          self.train_slide_path,
            'train_outcomes':       self.train_slide_record,
            'validation_paths':     self.test_slide_path,
            'validation_outcomes':  self.test_slide_record,
        }

        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
        with open('{1}/training_validation_testing_data{0}.json'.format(timestampStr, self.output_dir), 'w') as json_file:
            json.dump(d_trainsplit, json_file, indent=4, sort_keys=True)

    def NewResolution(self, image_size):
        '''
        Dispatches new image size to all ROIs in this dataset
        '''
        print ("Instructing all ROI transforms to down sample to " + str(image_size) + " pixels and update buffers to size " + str('NA'))
        for roi in self.train_slide_builders: roi.update_resolution_and_buffer(image_size)
        for roi in self.test_slide_builders: roi.update_resolution_and_buffer(image_size)


    def train(self):      self.mode = 'train'
    def build(self):      self.mode = 'build'
    def eval(self):       self.mode = 'eval'
    def interface(self):  self.mode = 'interface'

    def __len__(self):
        if   self.mode =='train':       return len(self.train_slide_builders)
        elif self.mode == 'build':      return len(self.all_builders)
        elif self.mode == 'eval':       return len(self.test_slide_builders)
        elif self.mode == 'interface':  return len(self.train_slide_builders)
        else: print ("Yikes!")

    def __getitem__(self, idx):
        torch.set_default_tensor_type('torch.FloatTensor')
        if self.mode == 'train':
            full_stack  = self.train_slide_builders[idx].get_train_data()
            outcome     = self.train_slide_builders[idx].params['outcome_tensor']
            return full_stack, outcome
        elif self.mode == 'build':
            full_stack  = self.all_builders[idx].build()
            outcome     = 0
            return full_stack, outcome
        elif self.mode == 'eval':
            full_stack  = self.test_slide_builders[idx].get_validation_data()
            outcome     = self.test_slide_builders[idx].params['outcome_tensor']
            return full_stack, outcome
        elif self.mode == 'interface':
            full_stack, cords, img_data  = self.all_builders[idx].get_inference_data()
            outcome     = self.all_builders[idx].params['outcome_tensor']
            filename    = self.all_builders[idx].params
            return full_stack, outcome, cords, filename
        else:
            print ("Yikes!")

if __name__ == "__main__":
    output_dir = '.'

    dataset = GHPSingleBagDatasetSimple(bag=True, output_dir=output_dir)
    dataset.load_new()


    dataset.train()
    dataset.NewResolution(256, 200)
    dataset.SetDropout(0.8)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=32, shuffle=True)
    print ("Testing")
    start = time.time()
    for batch_idx, master_batch_data in enumerate(dataloader):
        master_batch_data[0].cuda()
        for input_data in torch.split(master_batch_data[0].squeeze(0), 256):
            input_data.float()
            print(input_data.shape)
        print ("Done with batch ", batch_idx)
    end = time.time()

    dataset.eval()
    dataset.NewResolution(256, 200)
    dataset.SetDropout(0.8)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=32, shuffle=True)
    print ("Testing")
    start = time.time()
    for batch_idx, master_batch_data in enumerate(dataloader):
        master_batch_data[0].cuda()
        for input_data in torch.split(master_batch_data[0].squeeze(0), 256):
            input_data.float()
            print(input_data.shape)
        print ("Done with batch ", batch_idx)
    end = time.time()
    print (end-start)

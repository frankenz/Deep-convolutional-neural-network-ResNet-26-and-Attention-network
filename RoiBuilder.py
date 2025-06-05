import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from matplotlib import colors

import seaborn as sns
import numpy as np
import pandas as pd
import openslide

import sys
import glob
import cv2
import os
import random
import tifffile
import time

from PIL import Image
from PIL import ImageStat

class RoiBuilder():
    '''
     Class object that handles the organization, labeling, tile generation and cacheing of whole slide images.
     Usage:
        init(): Initialize the RoiBuilder.  Checks for cache files based on image name (must be unique). Some metadata is filled automatically.
                status will either be VALID or CACHE MISSING
        build(): Will trigger raster scan into tiles and L0 RoI identification based on pixel counts and statistics.
                MUST be called if status is CACHE MISSING before calling .generate() or .get() methods
                If successful status will be VALID
        update_resolution_and_buffer(): Define the image transforms, with output dimensions
                If successful, status will be set to VALID-READY
        generate(): Generate a 4D tensor of tiled images from the input WSI cache
                raises: RuntimeError if a cache or transform is missing

     Arguements:
        PATH_IMG: Full WSI path
        params: Dictonary of custom, user parameters (e.g. for caMIC), and outcome data (e.g. a outcome tensor, or list of outcomes per tile)
    '''
    # =============================================================================
    # Generic access functions
    def __init__(self, PATH_IMG, params):
        self.params = params

        # Base name should always be unique
        self.params['fullpath'] = PATH_IMG
        self.params['basename'] = os.path.split(PATH_IMG)[1].split('.')[0]
        self.params['root_cache_dir'] = os.path.expandvars('$CACHE_DIR')
        self.params['roi_size'] = 1200
        self.params['padding'] = 0
        self.params['ntiles'] = -1
        self.params['status'] = 'INIT'
        self.params['coor_cache'] = '{0}/coor_{1}_rois_size{2}_hsvcut_v3.npy'.format(self.params['root_cache_dir'] , self.params['basename'], self.params['roi_size'])
        self.params['data_cache'] = '{0}/data_{1}_rois_size{2}_hsvcut_v3.npy'.format(self.params['root_cache_dir'] , self.params['basename'], self.params['roi_size'])
        self.params['dropout'] = 0.9


        print ('-+-----------------------------------------------------------------------------------------------------------')
        print ("ROI builder for file at " + self.params['fullpath'])
        print (" | Checking....")

        self.loud = False

        if  (os.path.isfile(self.params['data_cache'])):
            raster = np.load(self.params['coor_cache'])
            print (" | ROI Builder is VALID with {0} tiles".format(len(raster)))
            self.params['ntiles'] = len(raster)
            self.params['status'] = 'VALID'
        else:
            print ("This will require building of ROIs.")
            self.params['status'] = 'CACHE MISSING'

        self.caMIC_eligable = False
        print ("Can this be opened in openside?")
        print (" | Checking...")
        try:
            openslide.OpenSlide(self.params['fullpath'])
            print (" | Yes!!")
            self.params['caMIC_eligable'] = True
        except:
            print (" | Nope!!")
            self.params['caMIC_eligable'] = False
        print ("RoiBuilder init() summary:")
        for key in self.params: print (" | {0:16s} = {1}".format(key, self.params[key]))
        print ('-+-----------------------------------------------------------------------------------------------------------')

    def getsize(self):
        '''
        Get the number of slide tiles
        '''
        return self.params['ntiles']

    def getname(self):
        '''
        The basename
        '''
        return self.params['basename']

    def getmeta(self):
        return self.params

    @staticmethod
    def sliding_window(dimensions, stepSize, padding):
        '''
        A static method for building a raster scan
        Paramters:
            dimensions: usually the image.shape of a tifffile
            stepSize: size of the ROI
            padding: region to ignore around the edges (useful for large SCN images with black boarders)
        '''
        raster = [ (x, y) for y in range(0 + padding, dimensions[1] - stepSize - padding - 1, stepSize) for x in range(0 + padding, dimensions[0] - stepSize - padding - 1, stepSize) ]
        return raster

    @staticmethod
    def array_read_region(arr, coord, downsample, size):
        '''
        Make a numpy array act like an openslide image, pull a tile from the whole array
        returns: raw_slice which is a uint8 numpy array and roi_slice which is a PIL image
        '''
        raw_slice = arr[coord[0] : coord[0] + size[0], coord[1] : coord[1] + size[1], :]
        roi_slice = Image.fromarray(arr[coord[0] : coord[0] + size[0], coord[1] : coord[1] + size[1], :])
        return roi_slice, raw_slice

    # =============================================================================
    # Builds cache
    def build(self):
        if 'VALID' in self.params['status']: return True

        # Double check?
        if os.path.isfile(self.params['data_cache']):
            data  = np.load(self.params['data_cache'])
            cords = np.load(self.params['coor_cache'])
            return True

        print ("Converting ROIs to numpy arrays ...")
        biggest_size = 0
        self.nseries = tifffile.TiffFile(self.params['fullpath']).series
        for i in range(0, len(self.nseries)):
            img = tifffile.imread(self.params['fullpath'], series=i)
            if np.prod(img.shape) > biggest_size:
                biggest_size = np.prod(img.shape)
                target_series = i
        img = tifffile.imread(self.params['fullpath'], series=target_series)

        print ("Selected series index (guessing to be 40x): ", target_series)
        print ("Selected series size  (guessing to be 40x): ", img.shape)

        roi_data   = []
        roi_coords = []

        raster = self.sliding_window(img.shape, self.params['roi_size'], padding=self.params['padding'])
        print ("Length of raster is " + str(len(raster)))

        for i, roi_coord in enumerate(raster):
            roi, data = self.array_read_region(img, roi_coord, 0, (self.params['roi_size'],self.params['roi_size']))
            # Check if there is contrast
            if ImageStat.Stat(roi).stddev[0] > 5    :

                h, s, v = cv2.split(np.asarray(roi.convert('HSV')))
                count = 0
                o = np.where(h > 120, 1, 0)
                o = np.where(v > 50,  o, 0)
                o = np.where(v < 210, o, 0)
                n_pass = np.sum(o)
                if n_pass > 1000:
                    roi_data.append(data)
                    roi_coords.append(roi_coord)

            print ( "Identified {0} ROIs so far and {1:.2f}% done with scan ... {2}".format( len(roi_coords), 100*(i+1)/len(raster), '\r' if not i+1==len(raster) else '\n'  ), end='' )

        np.save(self.params['data_cache'], roi_data)
        np.save(self.params['coor_cache'], roi_coords)
        self.params['status'] = 'VALID'

        return True


    # =============================================================================
    # In-training methods
    def update_resolution_and_buffer(self, resolution):
        '''
        Call this to update the resolution of the output tiles
        Important:
            Must be called before generation!
        Parameters:
            resolution: Integer <= ROI size
            buffer_size: Defunct
        '''

        if not 'VALID' in self.params['status']: raise RuntimeWarning("You're updating transforms for an uncached slide, call update_resolution_and_buffer() first")
        self.img_finalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(100),
            transforms.RandomCrop(self.params['roi_size']),
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
    #        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.img_finalize_flat = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.params['resolution'] = resolution
        self.params['status'] = 'VALID-READY'


    def get_train_data(self):
        '''
        Generate data for dataloader
        Tile randomization are in play
        Returns:
            [0] all_data which is a torch tensor of size [Ti, Nc, H, W]
        '''
        if not 'VALID-READY' in self.params['status']: raise RuntimeError("You're updating transforms for an uncached slide or haven't defined a transform, status = [{0}]")

        if  (os.path.isfile(self.params['data_cache'])):
            data  = np.load(self.params['data_cache'])
        else:
            raise RuntimeError("Somehow you initialized an RoiBuilder with no cache... quitting", self.params)

        # A hard limit...
        if data.shape[0] > 2500:
            data = data[np.random.choice(data.shape[0], 2500, replace=False)]

        # Only an issue for build new phase
        if len(data)==0:
            print ("No data found for ", self.params['basename'])
            return torch.zeros(20,3,128,128)

        return torch.stack([self.img_finalize(roi) for roi in data])

    def get_validation_data(self):
        '''
        Generate data for dataloader
        Tile randomization are in play
        Returns:
            [0] all_data which is a torch tensor of size [Ti, Nc, H, W]
        '''
        if not 'VALID-READY' in self.params['status']: raise RuntimeError("You're updating transforms for an uncached slide or haven't defined a transform, status = [{0}]")

        if  (os.path.isfile(self.params['data_cache'])):
            data  = np.load(self.params['data_cache'])
        else:
            raise RuntimeError("Somehow you initialized an RoiBuilder with no cache... quitting", self.params)

        # Only an issue for build new phase
        if len(data)==0:
            print ("No data found for ", self.params['basename'])
            return torch.zeros(20,3,128,128)

        return torch.stack([self.img_finalize_flat(roi) for roi in data])

    def get_inference_data(self):
        '''
        Generate data for dataloader
        No randomization or capping
        Useful for visualization, evaluation, and testing purposes, not recommended for training (where spatial information isn't used)
        Returns:
            [0] all_data which is a torch tensor of size [Ti, Nc, H, W]
            [1] matched ROI coordinates to the tensor index Ti
            [2] the origional numpy data (for imshow, usually)
        '''
        if not 'VALID-READY' in self.params['status']: raise RuntimeError("You're updating transforms for an uncached slide or haven't defined a transform, status = [{0}]")

        if  (os.path.isfile(self.params['data_cache'])):
            img_data  = np.load(self.params['data_cache'])
            cords = np.load(self.params['coor_cache'])
        else:
            print ("No cache file found... generating first")
            data  = np.load(self.params['data_cache'])
            cords = np.load(self.params['coor_cache'])

        data = [self.img_finalize_flat(roi) for roi in img_data]

        all_stacks = torch.stack( data )
        return all_stacks, cords, img_data

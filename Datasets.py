import torch
from torch.utils import data
from torchvision import transforms
from skimage.color import rgb2hed
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import glob
import cv2
import PyTorchHelpers
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])
import re
import random

import pandas as pd

from PIL import Image
from PIL import ImageStat



class CellImageDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.6])
        ])
        self.max_size = 128*128

        self.data_store = []

        print ("Let me just build a complete stockpile of data...")

        for img_name in glob.glob(self.root_dir  + '*wholecell-raw.png.jpg'):

            raw_image = cv2.imread(img_name)
            cell_mask = cv2.imread(img_name.replace('wholecell-raw.png.jpg','wholecell-mask.png'))
            nucl_mask = cv2.imread(img_name.replace('wholecell-raw.png.jpg','nucleus-mask.png'))

            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
            nucl_mask = cv2.cvtColor(nucl_mask, cv2.COLOR_BGR2GRAY)


            nucl_mask = cv2.bitwise_not(nucl_mask,nucl_mask)
            image = cv2.bitwise_and(raw_image, raw_image, mask=cell_mask)
            image = cv2.bitwise_and(image, image,   mask=nucl_mask)
            self.data_store.append(image)
            if len (self.data_store) > self.max_size: break


    def __len__(self):

        return self.max_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data_store[idx]

        if self.transform:
            image = self.transform(image)

        image = transforms.functional.adjust_contrast(image, 1)

        if self.finalize:
            image = self.finalize(image)
        sample = {'image': image}

        return sample


class CellImageDatasetHE(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.6])
        ])
        self.max_size = 64*256#len ( glob.glob(self.root_dir  + '*wholecell-raw.png*') )

        self.data_store = []
        self.data_raw = []
        self.coord_store = []

        print ("Let me just build a complete stockpile of data...")

        for img_name in glob.glob(self.root_dir  + '*wholecell-raw.png*'):
            if len (self.data_store) >= self.max_size: break

            image_string_coords = re.findall('(\d+[.]?\d+)',img_name.split("/")[-1])
            if len(image_string_coords) == 5:
                xcoord = int(image_string_coords[1]) + 0.5*int(image_string_coords[3])
                ycoord = int(image_string_coords[2]) + 0.5*int(image_string_coords[4])
            else:
                xcoord = 0
                ycoord = 0

            raw_image = cv2.imread(img_name)

            cv2_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            raw_image = img_as_float(cv2_image)

            ihc_hed = rgb2hed(raw_image)

            ihc_hed_f = np.float32(ihc_hed)

            ihc_hed_f_scaled = rescale_intensity(ihc_hed_f, in_range=(-.45,-.3), out_range=(0, 1))

            ihc_pil = Image.fromarray(ihc_hed_f_scaled[:,:,2], mode='F')

            raw_pil = Image.fromarray(cv2_image, mode='RGB')
            #
            # plt.subplot(2,3,1)
            # plt.imshow(raw_image)
            # plt.subplot(2,3,4)
            # plt.hist(raw_image.flatten())
            #
            # plt.subplot(2,3,2)
            # plt.imshow(ihc_hed_f[:,:,2], cmap=cmap_dab)
            # plt.subplot(2,3,5)
            # plt.hist(ihc_hed_f[:,:,2].flatten())
            #
            # plt.subplot(2,3,3)
            # plt.imshow(np.asarray(ihc_pil), cmap=cmap_dab)
            # plt.subplot(2,3,6)
            # plt.hist(np.asarray(ihc_pil).flatten())
            #
            # plt.show()

            self.data_raw.append(raw_pil)
            self.data_store.append(ihc_pil)
            self.coord_store.append(torch.FloatTensor([xcoord,ycoord]))

        print ("Loaded {0} images!!".format(len(self.data_raw)))

    def __len__(self):
        return len (self.data_store)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data_store[idx]
        raw_image = self.data_raw[idx]
        coord = self.coord_store[idx]

        image, raw_image = PyTorchHelpers.RandomCrop2X(64, pad_if_needed=True).Execute(image, raw_image)
        #raw_image = transforms.RandomCrop(64, pad_if_needed=True).Execute(image)
        if self.finalize:
            image = self.finalize(image)
            raw_image = self.finalize(raw_image)

        # Only get DAB channel
        sample = {'image': image, 'raw' : raw_image, 'xy' : coord}

        return sample




class CellImageDatasetRandomSpot(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, size=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.6])
        ])
        self.n_randomsamples = size#len ( glob.glob(self.root_dir  + '*wholecell-raw.png*') )

        self.data_store = []
        self.data_raw = []
        self.coord_store = []

        print ("Let me just build a complete stockpile of data...")

        for img_name in glob.glob(self.root_dir):

            raw_image = cv2.imread(img_name)

            cv2_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            raw_image = img_as_float(cv2_image)

            ihc_hed = rgb2hed(raw_image)

            ihc_hed_f = np.float32(ihc_hed)

            ihc_hed_f_scaled = rescale_intensity(ihc_hed_f, in_range=(-.45,-.3), out_range=(0, 1))

            ihc_pil = Image.fromarray(ihc_hed_f_scaled[:,:,2], mode='F')

            raw_pil = Image.fromarray(cv2_image, mode='RGB')

            self.data_raw.append(raw_pil)
            self.data_store.append(ihc_pil)

        print ("Loaded {0} images!!".format(len(self.data_raw)))

    def __len__(self):
        return self.n_randomsamples

    def __getitem__(self, idx):

        idx_spot = random.randint(0, len(self.data_raw)-1)
        image_FULL = self.data_store[idx_spot]
        raw_image_FULL = self.data_raw[idx_spot]

        cropper = PyTorchHelpers.RandomCrop2X(512, pad_if_needed=True)
        image, raw_image, coord = cropper.Execute(image_FULL, raw_image_FULL)
        while ImageStat.Stat(raw_image).stddev[0] < 5: image, raw_image, coord = cropper.Execute(image_FULL, raw_image_FULL)

        if self.finalize:
            image = self.finalize(image)
            raw_image = self.finalize(raw_image)

        return raw_image, image, coord







class IHCMixedBagDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, size=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.6])
        ])
        self.mini_batch_size = 128

        self.data_raw_rgb = []
        self.data_raw_dab = []
        self.data_raw_out = []

        self.coord_store = []

        print ("Let me just build a complete stockpile of data...")
        df = pd.read_csv('/home/andrew/Dropbox/DCPL/GBa2B-D-LC3AB-highPH-Cores/driver.csv',index_col=0)

        n_zero = 0
        n_ones = 0

        df = df.sample(frac=1, random_state=42)

        for index, row in df.iterrows():
            img_name = row['image_path']
            outcome = row['label']
            print ("Loading {0} labeled bag {1}".format(img_name,outcome))

            if outcome==1:
                n_ones+=1
                if n_ones > 2: continue
            if outcome==0:
                n_zero+=1
                if n_zero > 2: continue

            raw_image = cv2.imread(img_name)
            cv2_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            raw_image = img_as_float(cv2_image)
            ihc_hed = rgb2hed(raw_image)
            ihc_hed_f = np.float32(ihc_hed)
            ihc_hed_f_scaled = rescale_intensity(ihc_hed_f, in_range=(-.45,-.3), out_range=(0, 1))
            ihc_pil = Image.fromarray(ihc_hed_f_scaled[:,:,2], mode='F')
            raw_pil = Image.fromarray(cv2_image, mode='RGB')

            self.data_raw_rgb.append(raw_pil)
            self.data_raw_dab.append(ihc_pil)
            self.data_raw_out.append(torch.FloatTensor([outcome]))

        print ("Loaded {0} Bags!!".format(len(self.data_raw_rgb)))

    def __len__(self):
        return len(self.data_raw_rgb)

    def __getitem__(self, idx):

        FULL_RBG_IMAGE = self.data_raw_rgb[idx]
        FULL_DAB_IMAGE = self.data_raw_dab[idx]
        label = self.data_raw_out[idx]

        rgb_tiles = []
        dab_tiles = []
        coords = []
        labels = []
        cropper = PyTorchHelpers.RandomCrop2X(256, pad_if_needed=True)

        while len(rgb_tiles) < self.mini_batch_size:
            dab_tile, rgb_tile, coord = cropper.Execute(FULL_DAB_IMAGE, FULL_RBG_IMAGE)
            while ImageStat.Stat(rgb_tile).stddev[0] < 2: dab_tile, rgb_tile, coord = cropper.Execute(FULL_DAB_IMAGE, FULL_RBG_IMAGE)
            if self.finalize:
                dab_tile = self.finalize(dab_tile)
                rgb_tile = self.finalize(rgb_tile)
            rgb_tiles.append(rgb_tile)
            dab_tiles.append(dab_tile)
            coords.append(coord)
            labels.append(label)
        rbg_mini_batch = torch.stack(rgb_tiles)
        dab_mini_batch = torch.stack(dab_tiles)
        coords_mini_batch = torch.stack(coords)
        labels_mini_batch = torch.stack(labels)
        return rbg_mini_batch, dab_mini_batch, coords_mini_batch, labels_mini_batch

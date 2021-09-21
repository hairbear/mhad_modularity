# kinect_dataset.py
# Handong Wang 2021-09-15

import os

import cv2

import torch
from torch.utils.data import Dataset

# Gets a list of pgm files from the Kinect dataset, as well as loading precomputed Kinect features 
class KinectFeaturesDataset(Dataset):

    def __init__(self, directory):
        self.directory = directory
        entries = os.listdir(directory)
        files = [directory + entry for entry in entries if os.path.isfile(directory + entry)]
        self.pgms = []
        self.pts = []
        for file in files:
            filename, extension = os.path.splitext(file)
            if extension == ".pgm":
                self.pgms.append(file)
            elif extension == ".pt" and "features" in filename:
                self.pts.append(file)
        
        print("Kinect Dataset Length", len(self.pgms), len(self.pts))
        assert len(self.pgms) == len(self.pts) # Should be exactly 1 pytorch file corresponding to each pgm
        
        self.__len__ = len(self.pgms)
        self.pgms.sort()
        self.pts.sort()
        
        self.pgm_features = [torch.load(pt) for pt in self.pts]

    def __len__(self):
        return self.__len__

    def __getitem__(self, idx):
        return self.pgm_features[idx]

# Loads the pgms from the Kinect datasets as images
class KinectPgmsDataset(Dataset):

    def __init__(self, directory, background):
        self.directory = directory
        entries = os.listdir(directory)
        print("entries", entries)
        files = [directory + entry for entry in entries if os.path.isfile(directory + entry)]
        print("files", files)
        self.pgms = []
        self.pts = []
        for file in files:
            print("file", file)
            filename, extension = os.path.splitext(file)
            if extension == ".pgm":
                self.pgms.append(file)
            elif extension == ".pt":
                self.pts.append(file)
        print("phase 2")
        print("Kinect Dataset Length", len(self.pgms), len(self.pts))
        assert len(self.pgms) == len(self.pts) # Should be exactly 1 pytorch file corresponding to each pgm
        print("Kinect Dataset Length", len(self.pts))
        self.__len__ = len(self.pgms)
        self.pgms.sort()
        self.pts.sort()
        
        # load pgms as imgs
        self.imgs = [None for pgm in self.pgms]
        for pgm in self.pgms:
            image = cv2.imread(pgm)
            fg_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fg_gray[fg_gray == 0] = background[fg_gray == 0]
            dif_gray = cv2.absdiff(background, fg_gray)
            dif_gray = dif_gray[0:480, 0:480]
            # Arbitrary scaling up
            dif_gray *= 8
            retval, dif_gray = cv2.threshold(dif_gray, 255, 255, cv2.THRESH_TRUNC)
            dif = cv2.cvtColor(dif_gray, cv2.COLOR_GRAY2BGR)

    def __len__(self):
        return self.__len__

    def __getitem__(self, idx):
        return self.imgs[idx]
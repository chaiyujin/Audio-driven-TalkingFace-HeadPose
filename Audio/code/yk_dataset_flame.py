import copy
import os

# import utils
import pdb
import pickle
import random
import time
from glob import glob

import cv2
import numpy as np
import python_speech_features
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader

# EIGVECS = np.load('../basics/S.npy')
# MS = np.load('../basics/mean_shape.npy')


def find_clip_dirs(data_dir, with_train, with_test):
    # find clips
    clip_dirs = []
    for dirpath, subdirs, _ in os.walk(data_dir):
        is_trn = dirpath.find('/train/') >= 0
        is_tst = dirpath.find('/test/') >= 0
        if is_trn and not with_train:
            continue
        if is_tst and not with_test:
            continue
        # collect
        for subdir in subdirs:
            if subdir.startswith("clip") and os.path.exists(os.path.join(dirpath, subdir, "audio", "mfcc.npy")):
                clip_dirs.append(os.path.join(dirpath, subdir))
    clip_dirs = sorted(clip_dirs)
    return clip_dirs


class MultiClips_1D_lstm_3dmm(data.Dataset):
    def __init__(
        self,
        dataset_dir,
        train="train",
        relativeframe=0,
    ):
        self.train = train
        self.training = train == "train"
        self.num_frames = 16
        self.relativeframe = relativeframe

        self.data_list = []
        self.coordinates = []
        clip_dirs = find_clip_dirs(os.path.abspath(dataset_dir), self.training, not self.training)

        for i_clip, clip_dir in enumerate(clip_dirs):
            print(">> {}".format(clip_dir))

            # * MFCC
            mfcc_path = os.path.join(clip_dir, "audio", "mfcc.npy")
            mfcc = np.load(mfcc_path)
            mfccs = []
            ind = 3
            while ind <= int(mfcc.shape[0] / 4) - 4:
                # take 280 ms segment
                t_mfcc = mfcc[(ind - 3) * 4 : (ind + 4) * 4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
                ind += 1
            mfccs = torch.stack(mfccs, dim=0)
            # print(mfccs.shape)  # (24101, 28, 12) for Ben_Shapiro, (16282, 28, 12) for BBC_Carrie_Lam
            # self.mfccs = mfccs[self.start : self.start + self.trainN]

            # * Coefficients
            coeffc = np.load(os.path.join(clip_dir, "coeffs.npy"))
            coeffc2 = coeffc.copy()
            # self.coeffc = torch.FloatTensor(coeffc)
            # self.coeffc2 = torch.FloatTensor(coeffc2)

            # insert into list
            data_dict = dict(clip_dir=clip_dir)
            data_dict["mfccs"] = mfccs
            data_dict["coeffc"] = torch.FloatTensor(coeffc)
            data_dict["coeffc2"] = torch.FloatTensor(coeffc2)
            n_frames = min(mfccs.shape[0], coeffc.shape[0], coeffc2.shape[0])

            self.data_list.append(data_dict)
            for idx in range(0, n_frames - 16):
                self.coordinates.append((i_clip, idx))

    def __getitem__(self, i_item):
        i_clip, index = self.coordinates[i_item]
        data_dict = self.data_list[i_clip]
        coeffc = data_dict["coeffc"][index : index + 16]
        mfccs = data_dict["mfccs"][index : index + 16]
        coeffc2 = data_dict["coeffc2"][index : index + 16]
        return coeffc, mfccs, coeffc2

    def __len__(self):
        return len(self.coordinates)

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
#import PIL.Image
import json
import torch
import utils.dnnlib as dnnlib
import random
import pandas as pd
import glob
import soundfile as sf

import scipy.io
            
#try:
#    import pyspng
#except ImportError:
#    pyspng = None

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.
class InpaintingTestSet(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        num_samples=None,
        seed=42 ):
        


        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.test.path
        path_masks=dset_args.test.path_masks


        filelist=glob.glob(os.path.join(path,"*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"
        self.train_samples=filelist
        self.seg_len=int(seg_len)
        self.fs=fs

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        self.masks=[]

        for i in range(len(self.train_samples)):
                file=self.train_samples[i]
                self.filenames.append(os.path.basename(file))
                name=os.path.splitext(os.path.basename(file))[0]

                data, samplerate = sf.read(file)
                assert samplerate==self.fs, "error in dataloading: wrong samplerate"
                self._fs.append(samplerate)
                if len(data.shape)>1 :
                    data=np.mean(data,axis=1)
                #print(len(data), self.seg_len)
                data=data[0:self.seg_len]
                assert len(data)==self.seg_len, "error in dataloading: wrong length"
                self.test_samples.append(data) 

                try:
                    mask = scipy.io.loadmat(os.path.join(path_masks,name+'.mat'))["maskk"]
                except:
                    mask = scipy.io.loadmat(os.path.join(path_masks,'2_1759.mat'))["maskk"] #all the masks are the same
                mask=mask[0:self.seg_len]
                self.masks.append(mask)



    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self.masks[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


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

import utils.filter_generation_utils as f_utils

#try:
#    import pyspng
#except ImportError:
#    pyspng = None

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class LowpassFiltersData(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)


        self.args=dset_args
        NFFT=dset_args.NFFT

        #f=np.arange(0,fs/2,fs/NFFT)
        f=np.fft.rfftfreq(NFFT, 1/fs)
        f=torch.tensor(f) #shape (NFFT/2+1,)
        self.f=f.unsqueeze(0) #shape (1,NFFT/2+1)
        self.Nbatch=1 #for simplicity, maybe it is more efficient to do it in batch but then I should take into account how Iterabledataset works

        self.fs=fs

        self.freq_octs=torch.tensor(f_utils.get_third_octave_bands(fs, fmin=self.args.range.fmin, fmax=fs/2))
        self.freq_octs_idx=torch.argmin(torch.abs(self.f-self.freq_octs.unsqueeze(-1)), dim=-1)
        self.filter_length=self.freq_octs_idx.shape[-1]
        #H=f_utils.interpolate_filter(filter_octs, NFFT, fs,  freq_octs, interpolation="hermite_cubic")
        #print(H.shape)

        #A=10**(A/20)

    def randomize_parameters(self, decay_stages=1, fs=44100, fcmin=200, fcmax=8000, slope_min=-60, slope_max=0, N=1):
        fc1=np.exp(np.random.uniform(np.log(fcmin), np.log(fcmax), N))
        slope1=np.random.uniform(slope_min, slope_max, N)
        ret=[{'fc':torch.tensor(fc1), 'slope':torch.tensor(slope1)}]
        if decay_stages==2:
            fc2=np.exp(np.random.uniform(np.log(fc1), np.log(fcmax), N))
            slope2=np.random.uniform(slope_min, slope1, N)
       
            ret.append({'fc':torch.tensor(fc2), 'slope':torch.tensor(slope2)})
        return ret

    def __iter__(self):

        while True:
            A=torch.zeros((self.Nbatch,self.f.shape[-1])) #initialization of the filter

            random_filter_params=self.randomize_parameters(
                decay_stages=self.args.dist.decay_stages,
                fs=self.fs,
                N=self.Nbatch,
                fcmin=self.args.dist.fcmin, 
                fcmax=self.args.dist.fcmax,
                slope_min=self.args.dist.slope_min,
                slope_max=self.args.dist.slope_max
                )

            for params in random_filter_params:
                fc=params['fc'].unsqueeze(-1)
                slope=params['slope'].unsqueeze(-1)
                fc_idx=torch.argmin(torch.abs(self.f-fc), dim=-1)
                for i in range(fc_idx.shape[0]):
                    #I do that because I don't know how to do it in a vectorized way, or I am too lazy to do it
                    A[i,fc_idx[i]:]=slope[i]*torch.log2(self.f[:,fc_idx[i]:]/self.f[:,fc_idx[i]])+A[i,fc_idx[i]]
            
            yield A[:,self.freq_octs_idx]


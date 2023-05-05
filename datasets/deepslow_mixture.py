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

#try:
#    import pyspng
#except ImportError:
#    pyspng = None

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.
class DeepSlow_Mixture(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        overfit=False,
        stereo=False,
        seed=42 ):
        self.overfit=overfit

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path_ESC50=dset_args.path_ESC50
        path_freesound_loops=dset_args.path_freesound_loops
        path_LJSpeech=dset_args.path_LJSpeech

        filelistESC50=glob.glob(os.path.join(path_ESC50,"*.wav"))
        assert len(filelistESC50)>0 , "error in dataloading: empty or nonexistent folder"
        filelist_freesound=glob.glob(os.path.join(path_freesound_loops,"*.wav"))
        assert len(filelist_freesound)>0 , "error in dataloading: empty or nonexistent folder"
        filelist_LJSpeech=glob.glob(os.path.join(path_LJSpeech,"*.wav"))
        assert len(filelist_LJSpeech)>0 , "error in dataloading: empty or nonexistent folder"

        self.list_filelists=[filelistESC50,filelist_freesound,filelist_LJSpeech]

        #self.train_samples=filelist
        self.num_datasets=len(self.list_filelists)

        self.num_files_per_dataset=[len(filelist) for filelist in self.list_filelists]
       
        self.seg_len=int(seg_len)
        self.fs=fs
        self.stereo=dset_args.stereo
        self.normalize=dset_args.normalization.apply
        self.sigma_normalize=dset_args.normalization.sigma

        self.min_size=0.5 #minimum size of the audio in seconds, samples smaller than this are discarded
        if self.overfit:
            raise NotImplementedError
            #file=self.train_samples[0]
            #data, samplerate = sf.read(file)
            #if len(data.shape)>1 :
            #    data=np.mean(data,axis=1)
            #self.overfit_sample=data[10*samplerate:60*samplerate] #use only 50s

    def __iter__(self):
        if self.overfit:
           #data_clean=self.overfit_sample
           raise NotImplementedError
        while True:
            if not self.overfit:
                #choose random dataset
                dset_id=random.randint(0,self.num_datasets-1)
                good_sample=False
                while not good_sample:
                    #choose random file
                    num=random.randint(0,self.num_files_per_dataset[dset_id]-1)
                    #for file in self.train_samples:  
                    file=self.list_filelists[dset_id][num]
                    data, samplerate = sf.read(file)

                    if data.shape[-1]/samplerate > self.min_size:
                        good_sample=True

                #assert if samplerate is correct, if not print warning including the samplerate
                if samplerate!=self.fs:
                    #print("Warning: samplerate is not correct. Expected {}, got {}".format(self.fs,samplerate))
                    pass
                audio=data
                #Stereo to mono
                if not(self.stereo):
                    if len(data.shape)>1 :
                        audio=np.mean(audio,axis=1)
    
            #normalize
            #apply power normalization
            #if self.normalize:
            #    std=np.std(audio)
            #    audio=self.sigma_normalize*audio/std
            
            #framify data clean files
            #num_frames=np.floor(len(data_clean)/self.seg_len) 
            
            if not self.overfit:
                #print(audio.shape, self.seg_len)
                if audio.shape[-1]>=self.seg_len:
                    idx=np.random.randint(0,audio.shape[-1]-self.seg_len)
                    segment=audio[...,idx:idx+self.seg_len]
                else:
                    idx=0
                    #the length of the audio is smaller than the segment length
                    #just repeat the audio
                    segment=np.tile(audio,(self.seg_len//audio.shape[-1]+1))[...,idx:idx+self.seg_len]
            else:
                raise NotImplementedError("overfit not implemented")
                idx=0
                segment=audio[...,idx:idx+self.seg_len]

            segment=segment.astype('float32')
            #b=np.mean(np.abs(segment))
            #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                
            #let's make this shit a bit robust to input scale
            #scale=np.random.uniform(1.75,2.25)
            #this way I estimage sigma_data (after pre_emph) to be around 1
            
            #segment=10.0**(scale) *segment
            yield  segment, samplerate
            #else:
            #    pass



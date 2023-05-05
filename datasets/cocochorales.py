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
class AudioFolderDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        overfit=False,
        seed=42 ):
        self.overfit=overfit

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        self.dset_args=dset_args

        filelist=glob.glob(os.path.join(path,"*/"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"

        self.train_samples=filelist
       
        self.seg_len=int(seg_len)
        self.fs=fs
        if self.overfit:
            raise NotImplementedError
            file=self.train_samples[0]
            data, samplerate = sf.read(file)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)
            self.overfit_sample=data[10*samplerate:60*samplerate] #use only 50s
    def load_audio_file(self,file):
                data, samplerate = sf.read(file)
                assert(samplerate==self.fs, "wrong sampling rate")
                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    data_clean=np.mean(data_clean,axis=1)
                return data_clean

    def __iter__(self):
        if self.overfit:
           data_clean=self.overfit_sample
        while True:
            if not self.overfit:


                num=random.randint(0,len(self.train_samples)-1)

                #for file in self.train_samples:  
                file=self.train_samples[num]

                audio=np.zeros(self.seg_len)

                #get random number between 0 and 1
                rand_num=random.random()
                if rand_num<self.dset_args.prob_quartet:

                    #load the 4 stems
                    print("load 4 stems")
                    stems=glob.glob(os.path.join(file,"*.wav"))
                    if not( len(stems)==4):
                         "error in dataloading: wrong number of stems"
                    audio=[]
                    print(stems)
                    for s in stems:
                        audio+=[self.load_audio_file(s)]

                elif rand_num<self.dset_args.prob_quartet+self.dset_args.prob_trio:
                    #load 3 stems
                    print("load 3 stems")
                    stems=glob.glob(os.path.join(file,"*.wav"))
                    #assert len(stems)==4, "error in dataloading: wrong number of stems"
                    #remove one random stem
                    stems.pop(random.randrange(len(stems)))
                    #assert len(stems)==3, "error in dataloading: wrong number of stems"
                    print(stems)
                    audio=[]
                    for s in stems:
                        audio+=[self.load_audio_file(s)]

                elif rand_num<self.dset_args.prob_quartet+self.dset_args.prob_trio+self.dset_args.prob_duo:
                    #load 2 stems
                    print("load 2 stems")
                    stems=glob.glob(os.path.join(file,"*.wav"))
                    #assert len(stems)==4, "error in dataloading: wrong number of stems"
                    #remove two random stems
                    stems.pop(random.randrange(len(stems)))
                    stems.pop(random.randrange(len(stems)))
                    #assert len(stems)==2, "error in dataloading: wrong number of stems"
                    audio=[]
                    print(stems)
                    for s in stems:
                        audio+=[self.load_audio_file(s)]
                else:
                    #load 1 stem
                    print("load 1 stem")
                    stems=glob.glob(os.path.join(file,"*.wav"))
                    #assert len(stems)==4, "error in dataloading: wrong number of stems"
                    #remove two random stems
                    stems.pop(random.randrange(len(stems)))
                    stems.pop(random.randrange(len(stems)))
                    stems.pop(random.randrange(len(stems)))
                    #assert len(stems)==1, "error in dataloading: wrong number of stems"
                    audio=[]
                    print(stems)
                    for s in stems:
                        audio+=[self.load_audio_file(s)]

            #normalize
            #no normalization!!
            #data_clean=data_clean/np.max(np.abs(data_clean))
         
            #framify data clean files
            num_frames=np.floor(len(audio[0])/self.seg_len) 
            
            #if num_frames>4:
            for i in range(8):
                #get 8 random batches to be a bit faster
                if not self.overfit:
                    idx=np.random.randint(0,len(audio[0])-self.seg_len)
                else:
                    idx=0


                segment=audio[0][idx:idx+self.seg_len]
                if len(audio)>1:
                    for d in audio[1:]:
                        try:
                            segment+=d[idx:idx+self.seg_len]
                        except:
                            pass
                

                segment=segment.astype('float32')
                #b=np.mean(np.abs(segment))
                #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                    
                #let's make this shit a bit robust to input scale
                #scale=np.random.uniform(1.75,2.25)
                #this way I estimage sigma_data (after pre_emph) to be around 1
                
                #segment=10.0**(scale) *segment
                yield  segment
            #else:
            #    pass



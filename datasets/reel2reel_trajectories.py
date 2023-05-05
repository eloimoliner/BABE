
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
class ToyTrajectories(torch.utils.data.IterableDataset):
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
        orig_p=os.getcwd()
        os.chdir(path)
        filelist=glob.glob("*.wav")
        filelist=[os.path.join(path,f) for f in filelist]
        #print(filelist)
        os.chdir(orig_p)
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"

        self.train_samples=filelist
       
        self.seg_len=int(dset_args.seg_len)
        self.fs=dset_args.fs
        #print("overfit",self.overfit)

    def __iter__(self):
        while True:
            num=random.randint(0,len(self.train_samples)-1)
            #for file in self.train_samples:  
            file=self.train_samples[num]
            #print(file)
            data, samplerate = sf.read(file)
            #rint("loaded", data.shape)
            assert(samplerate==self.fs, "wrong sampling rate")
            data_clean=data
            #Stereo to mono
            if len(data.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
    
            #normalize
            #no normalization!!
            #data_clean=data_clean/np.max(np.abs(data_clean))
            #normalize mean
            #data_clean-=np.mean(data_clean, axis=-1)
            
         
            #framify data clean files
            num_frames=np.floor(len(data_clean)/self.seg_len) 
            
            #if num_frames>4:
            for i in range(8):
                #get 8 random batches to be a bit faster
                idx=np.random.randint(0,len(data_clean)-self.seg_len)
                segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                segment-=np.mean(segment, axis=-1)
                #b=np.mean(np.abs(segment))
                #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                    
                #let's make this shit a bit robust to input scale
                #scale=np.random.uniform(1.75,2.25)
                #this way I estimage sigma_data (after pre_emph) to be around 1
                
                #segment=10.0**(scale) *segment
                yield  segment
            #else:
            #    pass

class TestTrajectories(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        num_samples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.test.path

        print(path)
        #test_file=os.path.join(path,"input_3_.wav")
        orig_p=os.getcwd()
        os.chdir(path)
        filelist=glob.glob("*.wav")
        filelist=[os.path.join(path,f) for f in filelist]
        test_file=filelist[0]
        #print(filelist)
        os.chdir(orig_p)

        self.seg_len=int(seg_len)
        self.fs=fs

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        for i in range(num_samples):
            file=test_file
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data=data.T
            self._fs.append(samplerate)
            if data.shape[-1]>=self.seg_len:
                idx=np.random.randint(0,data.shape[-1]-self.seg_len)
                data=data[...,idx:idx+self.seg_len]
            else:
                idx=0
                data=np.tile(data,(self.seg_len//data.shape[-1]+1))[...,idx:idx+self.seg_len]
        

            #if not dset_args.test.stereo and len(data.shape)>1 :
            #   data=np.mean(data,axis=1)
            self.test_samples.append(data[...,0:self.seg_len]) #use only 50s


    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)



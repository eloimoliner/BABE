# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import re
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
import pickle
import timeout_decorator
from pytube import YouTube

#try:
#    import pyspng
#except ImportError:
#    pyspng = None

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class AudioSetTest(torch.utils.data.Dataset):
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

        filelist=glob.glob(os.path.join(path,"*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"
        self.train_samples=filelist
        self.seg_len=int(seg_len)
        self.fs=fs

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        for i in range(num_samples):
            file=self.train_samples[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data=data.T
            self._fs.append(samplerate)
            if data.shape[-1]>=self.seg_len:
                #idx=np.random.randint(0,data.shape[-1]-self.seg_len)
                idx=0
                data=data[...,idx:idx+self.seg_len]
            else:
                idx=0
                data=np.tile(data,(self.seg_len//data.shape[-1]+1))[...,idx:idx+self.seg_len]

            if not dset_args.stereo and len(data.shape)>1 :
                data=np.mean(data,axis=1)

            self.test_samples.append(data[...,0:self.seg_len]) #use only 50s


    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


class AudiosetStream(torch.utils.data.IterableDataset):
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
        path_balanced=dset_args.path_balanced_train_audio

        filelist_balanced=glob.glob(os.path.join(path_balanced,"*.wav"))
        #print(path_balanced)

        assert len(filelist_balanced)>0 , "error in dataloading: empty or nonexistent folder"

        self.balanced_train_samples=filelist_balanced
       
        self.seg_len=int(seg_len)
        self.fs=fs
        self.stereo=dset_args.stereo
        self.min_size=dset_args.min_size
        self.max_size_download=dset_args.max_size_download
        self.min_size_download=dset_args.ffmpeg_duration

        self.balanced_prob=dset_args.balanced_prob
        self.unbalanced_streams=glob.glob(os.path.join(dset_args.path_unbalanced_streams, "*.pkl"))

        self.tmp_path=dset_args.tmp_path
        self.ffmpeg_start=dset_args.ffmpeg_start
        self.ffmpeg_duration=dset_args.ffmpeg_duration
        self.samples_per_file=dset_args.samples_per_file

    #@timeout_decorator.timeout(3, timeout_exception=StopIteration)

    def load_from_balanced_set(self):
        num=random.randint(0,len(self.balanced_train_samples)-1)
        file=self.balanced_train_samples[num]
        good_sample=False
        while not good_sample: 
            audio, samplerate = sf.read(file)
            if audio.shape[0]/samplerate > self.min_size:
                    good_sample=True

        #print("balanced data", audio.shape, samplerate)
        #assert that data is stereo
        if not(self.stereo):
            if len(audio.shape)>1 :
                audio=np.mean(audio,axis=1)
        else:
            if len(audio.shape)==1:
                audio=np.expand_dims(audio,axis=0)
        audio=audio.T
        return audio, samplerate



    def load_from_unbalanced_set(self):
        good_sample=False
        while not good_sample:
            num=random.randint(0,len(self.unbalanced_streams)-1)
            with open(self.unbalanced_streams[num], 'rb') as f:
                stream_dic = pickle.load(f)
            """
            the stream_dic is a dictionary of tuples, they keys are the youtube ids, the values are:
                (stream, start_time, duration)
            """
    
            Ldic=len(stream_dic)
            num2=np.random.randint(0,Ldic)
            #select a random key
            key=list(stream_dic.keys())[num2]
            s=stream_dic[key]
            #p=self.download(s)
            duration=s[2]
            start=s[1]
            #print("duration", duration, "start", start)
            if duration>self.min_size_download and duration<self.max_size_download:
                try:
                    #print("downloading")
                    p= s[0].download(output_path=self.tmp_path) #most expensicve part
                    good_sample=True
                except Exception as e:
                    print("downloading failed", e)
                    good_sample=False

        #print("download done", p)

        name, ext=os.path.splitext(p)
        new_name=os.path.basename(name)
        new_name=re.sub('[^A-Za-z0-9]+', '', new_name)
        newp=os.path.join(self.tmp_path,new_name+ext)
        os.rename(p, newp)
        #move the file p to the new name newp

        audio_path=os.path.join(self.tmp_path,new_name+".wav")

        ffmpeg_stereo=2 if self.stereo else 1
        ffmpeg_fs=self.fs

        if start+duration>=self.ffmpeg_duration:
            ffmpeg_start=start
        else:
            ffmpeg_start=0
        

        #print(ffmpeg_start, self.ffmpeg_duration)
        #print(newp, audio_path)
        os.system("ffmpeg -y -loglevel error -i \"{}\" -ac {} -ar {} -t:00:00:{} {} {} "\
                            .format(newp, ffmpeg_stereo, ffmpeg_fs, ffmpeg_start, self.ffmpeg_duration,
                            audio_path)) #how much it takes depends on the duration
        #os.system("ffmpeg -y -i \"{}\" -ac {} -ar {} -t:00:00:{} {} {} "\
        #                    .format(p, ffmpeg_stereo, ffmpeg_fs, ffmpeg_start, self.ffmpeg_duration,
        #                    audio_path)) #how much it takes depends on the duration
        #print("ffmpeg done", audio_path)
        audio, samplerate=sf.read(audio_path) #how much it takes depends on the durantion
        os.remove(newp)
        os.remove(audio_path)
        #transpose to have the channels in the first dimension
        audio=audio.T
        return audio, samplerate


    def __iter__(self):
        while True:
            #decide if we load from balanced or unbalanced set
            if random.random()<=self.balanced_prob:
                #print("balanced")
                audio, samplerate=self.load_from_balanced_set()
                #print("audio shape balanced", audio.shape)

                if audio.shape[-1]>=self.seg_len:
                        idx=np.random.randint(0,audio.shape[-1]-self.seg_len)
                        segment=audio[...,idx:idx+self.seg_len]
                        segment=segment.astype('float32')
                        #print("yielding balanced", segment.shape, samplerate)
                        yield  segment, samplerate
                else:
                        idx=0
                        #the length of the audio is smaller than the segment length
                        #just repeat the audio
                        segment=np.tile(audio,(self.seg_len//audio.shape[-1]+1))[...,idx:idx+self.seg_len]
                        segment=segment.astype('float32')
                        #print("yielding balanced small", segment.shape, samplerate)
                        yield  segment, samplerate
               
            else:
                print("unbalanced")
                audio, samplerate= self.load_from_unbalanced_set()
               
                #print("audio shape unbalanced", audio.shape)

                if audio.shape[-1]>=self.seg_len:
                    for i in range(self.samples_per_file):
                        idx=np.random.randint(0,audio.shape[-1]-self.seg_len)
                        segment=audio[...,idx:idx+self.seg_len]
                        segment=segment.astype('float32')
                        #print("yielding unbalanced", segment.shape, samplerate)
                        yield  segment, samplerate
               
                else:
                        #just repeat the audio and yield 1
                        idx=0
                        #the length of the audio is smaller than the segment length
                        #just repeat the audio
                        segment=np.tile(audio,(self.seg_len//audio.shape[-1]+1))[...,idx:idx+self.seg_len]
               
                        segment=segment.astype('float32')
                        #print("yielding small", segment.shape, samplerate)
                        yield  segment, samplerate




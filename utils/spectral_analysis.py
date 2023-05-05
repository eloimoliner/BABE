
import plotly.express as px
import torch
from datasets.maestro_dataset import MaestroDataset_fs
import torchaudio
from omegaconf import OmegaConf
import math


class LTAS_processor():

    def __init__(self, sample_rate, audio_len ):
        self.sample_rate=sample_rate
        self.audio_len=audio_len
    def compute_fft(self, audio):
        #compute fft
        FFT_audio=torch.fft.rfft(audio, norm="forward")
        FFT_audio=torch.abs(FFT_audio)**2
        #in dB
        FFT_audio=10*torch.log10(FFT_audio)
        return FFT_audio
    def resample(self, audio, fs):
        if fs!=self.sample_rate:
            audio=torchaudio.functional.resample(audio, fs, self.sample_rate)
        return audio

    def measure_LTAS(self, dataloader, num_samples=1000):
        self.LTAS=torch.zeros(self.audio_len//2+1)
    
        num=0
        for i in range(1000):
            audio, fs=next(dataloader)
            #resample
            audio=self.resample(audio, fs)
            audio=audio[..., :self.audio_len]
            audio=audio.squeeze(0)
            print(i)
            #compute fft
            FFT_audio=self.compute_fft(audio)
            print(FFT_audio.shape)
            num+=1
            self.LTAS=self.LTAS*(1-1/num)+(FFT_audio)/num
        torch.save(self.LTAS, "LTAS.pt")
        return self.LTAS


    def load_dataset_LTAS(self, file):
        self.LTAS=torch.load(file)

    def plot_LTAS(self, plot_LTAS=None):
        if plot_LTAS is None:
            plot_LTAS=self.LTAS

        fig=px.line(x=torch.linspace(0, self.sample_rate/2, self.audio_len//2+1), y=plot_LTAS, log_x=True)
        #set limits of x axis (they are in log scale)
        fig.update_xaxes(range=[1.3, 4.3])
        #set limits of y axis
        #save the plot as html
        fig.write_html("LTAS.html")
        return fig
    def compute_gaussian_window(self, band, Noct=1):
        sigma = (band/Noct)/math.pi
        gaussian_window=torch.exp(-0.5*((torch.linspace(0, self.audio_len//2, self.audio_len//2+1)-band*self.audio_len/self.sample_rate)/(sigma*self.audio_len/self.sample_rate))**2)
        return gaussian_window

    def rescale_audio_to_LTAS(self, audio,fs, band=500):
        if fs!=self.sample_rate:
            audio=self.resample(audio, fs)
        #compute fft
        FFT_audio=self.compute_fft(audio)
        #apply bandpass filter to FFT_audio
        #design a gaussian window in the frequency domain, centered in the middle of the band and with a width of a octave  band
        #inverse fft
        gaussian_window=self.compute_gaussian_window(band)
        self.LTAS=self.LTAS.to(audio.device)
        gaussian_window=gaussian_window.to(audio.device)

        LTAS_linear=10**(self.LTAS/20)
        windowed_LTAS=LTAS_linear*gaussian_window
        sum_LTAS=torch.sum(windowed_LTAS)

        FFT_audio_linear=10**(FFT_audio/20)
        windowed_FFT_audio=FFT_audio_linear*gaussian_window
        sum_FFT_audio=torch.sum(windowed_FFT_audio)
        difference=20*torch.log10(sum_LTAS/sum_FFT_audio)

        #print("windowed_LTAS", torch.mean(windowed_LTAS), "dB")
        #print("windowed_FFT_audio", torch.mean(windowed_FFT_audio), "dB")

        #print("difference", difference, "dB")
        #rescaled_audio=audio*10**(difference/20)

        return  difference



def main():
    dset_args={
        "path":"/scratch/shareddata/dldata/maestro/v3.0.0/maestro-v3.0.0",
        "years": [2004,2006,2008,2009,2011, 2013, 2014, 2015, 2017, 2018],
        "load_len": 405000
    }

    dset_args=OmegaConf.create(dset_args)

    sample_rate=44100
    audio_len=368368


    dataset=MaestroDataset_fs(dset_args, overfit=False)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=1,  num_workers=1, pin_memory=True,  timeout=0,))

    LTAS=torch.zeros(audio_len//2+1)

    num=0
    for i in range(10000):
        audio, fs=next(dataset_iterator)
        #resample
        if fs!=sample_rate:
            audio=torchaudio.functional.resample(audio, fs, sample_rate)
        audio=audio[..., :audio_len]
        audio=audio.squeeze(0)
        print(i)
        #compute fft
        FFT_audio=torch.fft.rfft(audio, norm="forward")
        FFT_audio=torch.abs(FFT_audio)**2
        #in dB
        FFT_audio=10*torch.log10(FFT_audio)
        print(FFT_audio.shape)
        num+=1
        LTAS=LTAS*(1-1/num)+(FFT_audio)/num

    #save the LTAS
    torch.save(LTAS, "LTAS.pt")
    
    #plot the LTAS with plotly
    fig=px.line(x=torch.linspace(0, sample_rate/2, audio_len//2+1), y=LTAS, log_x=True)
    #set limits of x axis (they are in log scale)
    fig.update_xaxes(range=[1.3, 4.3])
    #set limits of y axis
    #save the plot as html
    fig.write_html("LTAS.html")


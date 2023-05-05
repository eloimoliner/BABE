import os
import sys
from frechet_audio_distance import FrechetAudioDistance
import torch
import soundfile as sf
import glob
from tqdm import tqdm

def get_LSD_of_set(test, bg):
    refs=glob.glob(os.path.join(bg,"*.wav"))

    LSD_sum=0
    LSD_count=0

    for ref in tqdm(refs):

        shit, name=os.path.split(ref)

        testfile=os.path.join(test, name)
        assert os.path.exists(testfile), "file not found: "+testfile

        audio_ref,fs =sf.read(ref)
        audio_test,fs2 =sf.read(testfile)

        #assert fs2==fs
        assert audio_ref.shape==audio_test.shape
         

        audio_ref=torch.Tensor(audio_ref)
        audio_test=torch.Tensor(audio_test)

        LSD=computeLSD(audio_ref.unsqueeze(0), audio_test.unsqueeze(0))

        LSD_sum += LSD
        LSD_count += 1

    LSD_sum/=LSD_count
    return LSD_sum


def computeLSD(y,y_g): #check!!!
    yF=do_stft(y,win_size=2048, hop_size=512)
    ygF=do_stft(y_g, win_size=2048, hop_size=512)
    yF=torch.sqrt(yF[:,0,:,:]**2 +yF[:,1,:,:]**2)
    ygF=torch.sqrt(ygF[:,0,:,:]**2 +ygF[:,1,:,:]**2)
    Sy = torch.log10(torch.abs(yF)**2 + 1e-8)
    Syg = torch.log10(torch.abs(ygF)**2 + 1e-8)
    lsd = torch.sum(torch.mean(torch.sqrt(torch.mean((Sy-Syg)**2 + 1e-8, axis=2)), axis=1), axis=0)
    return lsd

def do_stft(noisy, win_size=2048, hop_size=512, device="cpu"):

    #window_fn = tf.signal.hamming_window

    #win_size=args.stft.win_size
    #hop_size=args.stft.hop_size
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)

    return stft_signal_noisy


#path=sys.argv[1]
bg=sys.argv[1]
test=sys.argv[2]
#print(path)
#print(bg)
#print(test)
print("Computing LSD of: ",test, " using background: ",bg)
LSD=get_LSD_of_set(test,bg)
print("Result LSD: ",LSD)

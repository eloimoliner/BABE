import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import math as m
import torch
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

from cqt_nsgt_pytorch import CQT_nsgt
import torchaudio

"""
As similar as possible to the original CQTdiff architecture, but using the octave-base representation of the CQT
This should be more memory efficient, and also more efficient in terms of computation, specially when using higher sampling rates.
I am expecting similar performance to the original CQTdiff architecture, but faster. 
Perhaps the fact that I am using powers of 2 for the time sizes is critical for transient reconstruction. I should thest CQT matrix model with powers of 2, this requires modifying the CQT_nsgt_pytorch.py file.
"""
class BiasFreeGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-7, min_channels_per_group=4):
        super(BiasFreeGroupNorm, self).__init__()

        self.num_groups = min(num_groups, num_features // min_channels_per_group)
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        #self.beta = torch.zeros(1,num_features,1,1)
        #self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups ,-1)
        #mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        # normalize
        x = (x) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.gamma
class CombinerUp(nn.Module):
    """
    Combining after upsampling in the decoder side, using progressive growing at the style of stylegan2
    """

    def __init__(self, Npyr, Nx, bias=True):
        """
        Args:
            Npyr (int): Number of channels of the pyramidal signal to upsample (usually 2)
            Nx (int): Number of channels of the latent vector to combine
        """
        super().__init__()
        self.conv1x1=nn.Conv2d(Nx, Npyr,1, bias=bias)
        #self.GN=nn.GroupNorm(8,Nx)
        torch.nn.init.constant_(self.conv1x1.weight, 0)
    def forward(self,pyr,x):
        """
        Args:
            pyr (Tensor): shape (B,C=2,F,T) pyramidal signal 
            x (Tensor): shape (B,C,F,T)  latent 
        Returns:
           Rensor with same shape as x
        """
                
        x=self.conv1x1(x)
        if pyr==None:
            return x
        else:
            
            return (pyr[...,0:x.shape[-1]]+x)/(2**0.5)

class CombinerDown(nn.Module):
    """
    Combining after downsampling in the encoder side, with progressive growing at the style of stylegan2
    """

    def __init__(self, Nin, Nout, bias=True):
        """
        Args:
            Npyr (int): Number of channels of the pyramidal signal to downsample (usually 2)
            Nx (int): Number of channels of the latent vector to combine
        """
        super().__init__()
        self.conv1x1=nn.Conv2d(Nin, Nout,1, bias=bias)

    def forward(self,pyr,x):
        """
        Args:
            pyr (Tensor): shape (B,C=2,F,T) pyramidal signal 
            x (Tensor): shape (B,C,F,T)  latent 
        Returns:
            Tensor with same shape as x
        """
        pyr=self.conv1x1(pyr)
        return (pyr+x)/(2**0.5)

class Upsample(nn.Module):
    """
        Upsample time dimension using resampling
    """
    def __init__(self,S):
        """
        Args:
            S (int): upsampling factor (usually 2)
        """
        super().__init__()
        N=2**12
        self.resample=torchaudio.transforms.Resample(N,N*S) #I use 3**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        #self.resample=nn.Upsample( scale_factor=(1,S))
    def forward(self,x):
        return self.resample(x) 

class Downsample(nn.Module):
    """
        Downsample time dimension using resampling
    """
    def __init__(self,S):
        """
        Args:
            S (int): downsampling factor (usually 2)
        """
        super().__init__()
        N=2**12
        self.resample=torchaudio.transforms.Resample(N,N/S) #I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        #self.resample=nn.AvgPool2d((1,S+1), stride=(1,S), padding=(0,1))
    def forward(self,x):
        return self.resample(x) 
    

class RFF_MLP_Block(nn.Module):
    """
        Encoder of the noise level embedding
        Consists of:
            -Random Fourier Feature embedding
            -MLP
    """
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class AddFreqEncodingRFF(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim, N):
        super(AddFreqEncodingRFF, self).__init__()
        self.N=N
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, N]), requires_grad=False)


        self.f_dim=f_dim #f_dim is fixed
        embeddings=self.build_RFF_embedding()
        self.embeddings=nn.Parameter(embeddings, requires_grad=False) 

        
    def build_RFF_embedding(self):
        """
        Returns:
          table:
              (shape: [C,F], dtype: float32)
        """
        freqs = self.RFF_freq
        #freqs = freqs.to(device=torch.device("cuda"))
        freqs=freqs.unsqueeze(-1) # [1, 32, 1]

        self.n=torch.arange(start=0,end=self.f_dim)
        self.n=self.n.unsqueeze(0).unsqueeze(0)  #[1,1,F]

        table = 2 * np.pi * self.n * freqs

        #print(freqs.shape, x.shape, table.shape)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) #[1,32,F]

        return table
    

    def forward(self, input_tensor):

        #print(input_tensor.shape)
        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[-1]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.embeddings, [batch_size_tensor, time_dim,self.N*2, self.f_dim])
        fembeddings_2=fembeddings_2.permute(0,2,3,1)
    
        
        #print(input_tensor.shape, fembeddings_2.shape)
        return torch.cat((input_tensor,fembeddings_2),1)  

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

def orthogonal_(module):
    nn.init.orthogonal_(module.weight)
    return module

class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(orthogonal_(nn.Linear(feats_in if i == 0 else feats_out, feats_out)))
            layers.append(nn.GELU())
        super().__init__(*layers)

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        dilation=(1,1),
        use_atention=False,
        attention_dict=None, #TODO when implementing attention include here the hyperparameters
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        #self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head

        self.norm0 = BiasFreeGroupNorm(in_channels, eps=1e-8)

        #I think this should be a 1x1 conv or local conv. This is also where downsampling should happen. Local is safer
        if up:
            self.conv0 = nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel_size=3, bias=False , padding="same"),UpDownResample(up=True))
        elif down:
            self.conv0 = nn.Sequential(UpDownResample(down=True), nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding="same"))
        else:
            self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding="same")
        #I think this is the most efficient way to do it.  1x1 conv here


        #basically film
        self.affine = nn.Linear(emb_channels, out_channels) #take care of init!

        self.norm1 = BiasFreeGroupNorm(out_channels, eps=1e-8)
        #this is a dilated conv
        self.conv1 = nn.Conv2d(out_channels, out_channels, (5,5), dilation=dilation, padding="same", bias=False)

        self.skip_scale=1/(2**0.5)


        skip_resample=nn.Identity()
        skip_project=nn.Identity()
        if up or down:
            skip_resample=UpDownResample(up=up,down=down)
        if out_channels != in_channels or up or down:
            skip_project = nn.Conv2d(in_channels,out_channels, kernel_size=1, bias=False, padding="same")

        self.skip = nn.Sequential(skip_resample,skip_project)

        self.use_attention=use_atention
        if self.use_attention:
            pass
            #self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            #self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            #self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)
        

    def forward(self, x, emb):
        orig = x
        #print(x.shape)
        x = self.conv0(F.gelu(self.norm0(x)))

        scale = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        #AdaGN without shift
        x = F.gelu(torch.mul(self.norm1(x), scale + 1)) #the +1 is meant to induce values close to 0 to the output of the linear layer
        #I wont use droput for now, too lazy to implement it
        #x=torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x) #this is a 1x1 conv or local conv. Maybe local 3x3 is safer

        x = (x+ self.skip(orig))*self.skip_scale

        if self.use_attention:
            
            #TODO implement attention
            #q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            #w = AttentionOp.apply(q, k)
            #a = torch.einsum('nqk,nck->ncq', w, v)
            #x = self.proj(a.reshape(*x.shape)).add_(x)
            #x = x * self.skip_scale
            pass

        return x

class Film(nn.Module):
    def __init__(self, output_dim, bias=True):
        super().__init__()
        self.bias=bias
        if bias:
            self.output_layer = nn.Linear(512, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(512, 1 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        sigma_encoding = sigma_encoding.unsqueeze(-1) #we need a secnond unsqueeze because our data is 2d [B,C,1,1]
        if self.bias:
            gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        else:
            gamma=sigma_encoding
            beta=None

        return gamma, beta

class Gated_residual_layer(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        dilation,
        bias=True
    ):
        super().__init__()
        self.conv= nn.Conv2d(dim,dim,
                  kernel_size=kernel_size,
                  dilation=dilation,
                  stride=1,
                  padding='same',
                  padding_mode='zeros', bias=bias) #freq convolution (dilated) 
        self.act= nn.GELU()
        #self.conv1_1= nn.Conv2d(dim,dim,
        #          kernel_size=1)
        #self.position_gate = nn.Sequential( nn.Linear(64, dim),
        #                                    nn.Sigmoid()) #assuming that 64 is the dimensionality of the RFF freq. positional embeddings
        #self.gn=nn.GroupNorm(8, dim)

    def forward(self, x):
        #gate=self.position_gate(freqembeddings)  #F, N
        #B,N,T,F=x.shape
        #gate = gate.unsqueeze(0).unsqueeze(0) #1,1, F,N
        #gate = gate.permute(0,3,2,1) #1,N,1, F
        #torch.broadcast_to(gate.permute(0,1).unsqueeze(0).unsqueeze(0)
        
        x=(x+self.conv(self.act(x)))/(2**0.5)
        return x
        
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=False,
        num_dils = 6,
        bias=True,
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.film=Film(dim, bias=bias)

        self.res_conv = nn.Conv2d(dim, dim_out, 1, padding_mode="zeros", bias=bias) if dim != dim_out else nn.Identity()

        self.H=nn.ModuleList()
        self.num_dils=num_dils

        if self.use_norm:
            self.gnorm=nn.GroupNorm(8,dim)

        self.first_conv=nn.Sequential(nn.GELU(),nn.Conv2d(dim, dim_out,1, bias=bias))

         
        for i in range(self.num_dils):
            self.H.append(Gated_residual_layer(dim_out, (5,3), (2**i,1), bias=bias)) #sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)


    def forward(self, x, sigma):
        
        gamma, beta = self.film(sigma)

        if self.use_norm:
            x=self.gnorm(x)

        if self.bias:
            x=x*gamma+beta
        else:
            x=x*gamma #no bias

        y=self.first_conv(x)

        
        for h in self.H:
            y=h(y)

        return (y + self.res_conv(x))/(2**0.5)

_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic': 
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3': 
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
class UpDownResample(nn.Module):
    def __init__(self,
        up=False, 
        down=False,
        mode_resample="T", #T for time, F for freq, TF for both
        resample_filter='cubic', 
        pad_mode='reflect'
        ):
        super().__init__()
        assert not (up and down) #you cannot upsample and downsample at the same time
        assert up or down #you must upsample or downsample
        self.down=down
        self.up=up
        if up or down:
            #upsample block
            self.pad_mode = pad_mode #I think reflect is a goof choice for padding
            if mode_resample=="T":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            else:
                raise NotImplementedError("Only time upsampling is implemented")
                #TODO implement freq upsampling and downsampling
            self.pad = kernel_1d.shape[0] // 2 - 1
            self.register_buffer('kernel', kernel_1d)
    def forward(self, x):
        shapeorig=x.shape
        #x=x.view(x.shape[0],-1,x.shape[-1])
        x=x.view(-1,x.shape[-2],x.shape[-1])
        if self.down:
            x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        elif self.up:
            x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)

        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        if self.down:
            x_out= F.conv1d(x, weight, stride=2)
        elif self.up:
            x_out =F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)

        return x_out.view(shapeorig[0],-1,shapeorig[2], x_out.shape[-1])


class Unet_CQT_oct(nn.Module):
    """
        Main U-Net model based on the CQT
    """
    def __init__(self, args, device):
        """
        Args:
            args (dictionary): hydra dictionary
            device: torch device ("cuda" or "cpu")
        """
        super(Unet_CQT_oct, self).__init__()
        self.args=args
        self.depth=args.network.depth
        #self.embedding = RFF_MLP_Block()
        self.use_norm=args.network.use_norm

        emb_channels=args.network.emb_channels# 256 or 512 add to the config
        self.sigma_embed = FourierFeatures(1, emb_channels)
        self.mapping = MappingNet(emb_channels,emb_channels)

        #fmax=self.args.exp.sample_rate/2
        #self.fmin=fmax/(2**self.args.cqt.numocts)
        self.fbins=int(self.args.network.cqt.bins_per_oct*self.args.network.cqt.num_octs) 
        self.device=device
        self.bins_per_oct=self.args.network.cqt.bins_per_oct
        self.num_octs=self.args.network.cqt.num_octs
        self.CQTransform=CQT_nsgt(self.args.network.cqt.num_octs,self.args.network.cqt.bins_per_oct, "oct",  self.args.exp.sample_rate, self.args.exp.audio_len, device=self.device)

        assert self.depth==self.args.network.cqt.num_octs, "depth should be equal to the number of octaves in the CQT"

        self.f_dim=self.fbins #assuming we have thrown away the DC component and the Nyquist frequency

        self.use_fencoding=self.args.network.use_fencoding
        if self.use_fencoding:
            N_freq_encoding=32
    
            self.freq_encodings=nn.ModuleList([])
            for i in range(self.num_octs):
                self.freq_encodings.append(AddFreqEncodingRFF(self.bins_per_oct,N_freq_encoding))
            Nin=2*N_freq_encoding+2
        else:
            Nin=2

        #Encoder
        self.Ns= self.args.network.Ns
        self.Ss= self.args.network.Ss
        self.num_dils= self.args.network.num_dils #intuition: less dilations for the first layers and more for the deeper layers
        
        self.init_conv= nn.Conv2d(Nin,self.Ns[0],(5,3), padding="same", padding_mode="zeros", bias=False)

        self.downsampler=UpDownResample(down=True)
        self.upsampler=UpDownResample(up=True)


        self.downs=nn.ModuleList([])
        self.middle=nn.ModuleList([])
        self.ups=nn.ModuleList([])
        
        skips = []

        for i in range(self.depth):
            if i==0:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i-1]
                dim_out=self.Ns[i]

            list_of_blocks=nn.ModuleList([])
            dim_in_b=dim_in
            for j in range(self.num_dils[i]):
                list_of_blocks.append(UNetBlock(dim_in_b, dim_out, emb_channels, dilation=(2**(j),1)))
                dim_in_b=dim_out

            skips.append(dim_out)
            self.downs.append(
                               nn.ModuleList([
                                        nn.Conv2d(Nin, dim_in, (5,3), padding="same", padding_mode="zeros", bias=False),
                                        nn.Conv2d(2, dim_out, (5,3), padding="same", padding_mode="zeros", bias=False),
                                        list_of_blocks]
                                        ))


            
        self.bottleneck=nn.ModuleList([])
        for j in range(self.num_dils[-1]):
            dim=self.Ns[-1]
            self.bottleneck.append(UNetBlock(dim, dim, emb_channels, dilation=(2**(j),1)))

                        

        self.pyr_up_proj_first=nn.Conv2d(dim_out, 2, (5,3), padding="same", padding_mode="zeros", bias=False)
        for i in range(self.depth-1,-1,-1):

            if i==0:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i-1]

            list_of_blocks=nn.ModuleList([])
            dim_in_b=dim_in+skips.pop()
            for j in range(self.num_dils[i]):
                list_of_blocks.append(UNetBlock(dim_in_b, dim_out, emb_channels, dilation=(2**(j),1)))
                dim_in_b=dim_out


            self.ups.append(nn.ModuleList(
                                        [
                                        nn.Conv2d(dim_out, 2, (5,3), padding="same", padding_mode="zeros", bias=False),
                                        list_of_blocks]
                                       ))



        self.cropconcat = CropConcatBlock()




    def forward(self, inputs, c_noise):
        """
        Args: 
            inputs (Tensor):  Input signal in time-domsin, shape (B,T)
            sigma (Tensor): noise levels,  shape (B,1)
        Returns:
            pred (Tensor): predicted signal in time-domain, shape (B,T)
        """
        #apply RFF embedding+MLP of the noise level
        #sigma = self.embedding 
        sigma = self.sigma_embed(c_noise)
        #TODO add extra embedding for the frequency
        sigma = self.mapping(sigma)

        
        #apply CQT to the inputs
        X_list =self.CQTransform.fwd(inputs.unsqueeze(1))
        X_list_out=X_list

        hs=[]
        for i,modules in enumerate(self.downs):
            C=X_list[-1-i]#get the corresponding CQT octave
            C=C.squeeze(1)
            C=torch.view_as_real(C)
            C=C.permute(0,3,1,2).contiguous() # call contiguous() here?
            if self.use_fencoding:
                #Cfreq=self.freq_encoding(C)
                Cfreq=self.freq_encodings[i](C) #B, C + Nfreq*2, F,T
                
            init_proj, pyr_down_proj, ResBlocks=modules

            Cfreq=init_proj(Cfreq)
            
            if i==0:
                X=Cfreq #starting the main signal path
                pyr=self.downsampler(C) #starting the auxiliary path
            elif i<(self.depth-1):
                pyr=torch.cat((self.downsampler(C),self.downsampler(pyr)),dim=2) #updating the auxiliary path
                X=torch.cat((Cfreq,X),dim=2) #updating the main signal path with the new octave
            else:# last layer
                X=torch.cat((Cfreq,X),dim=2) #updating the main signal path with the new octave

            for h in ResBlocks:
                X=h(X,sigma)
            hs.append(X)

            #downsample the main signal path
            #we do not need to downsample in the inner layer
            if i<(self.depth-1): 
                X=self.downsampler(X)

                #apply the residual connection
                X=(X+pyr_down_proj(pyr))/(2**0.5) #I'll my need to put that inside a combiner block??
            #print("encoder ", i, X.shape)
                
        #middle layers
        #(ResBlock,)=self.middle[0]
        for h in self.bottleneck:
            X=h(X,sigma)
        #X=ResBlock(X, sigma)   

        pyr=self.pyr_up_proj_first(X)
        for i,modules in enumerate(self.ups):
            j=self.depth -i-1
            pyr_up_proj, ResBlocks=modules

            skip=hs.pop()
            X=torch.cat((X,skip),dim=1)
            for h in ResBlocks:
                X=h(X,sigma)
            #X=ResBlock(X, sigma)

            X_out, X= X[:,:,0:self.bins_per_oct,:], X[:,:,self.bins_per_oct::,:]
            pyr_out, pyr= pyr[:,:,0:self.bins_per_oct,:], pyr[:,:,self.bins_per_oct::,:]

            X_out=(pyr_up_proj(X_out)+pyr_out)/(2**0.5)
            X_out=X_out.permute(0,2,3,1).contiguous() #call contiguous() here?
            X_out=torch.view_as_complex(X_out)

            #save output
            X_list_out[i]=X_out.unsqueeze(1)

            if j>0: 
                pyr=self.upsampler(pyr) #call contiguous() here?
                X=self.upsampler(X) #call contiguous() here?

            #print("decoder ", i, X.shape)

        pred_time=self.CQTransform.bwd(X_list_out)
        pred_time=pred_time.squeeze(1)
        pred_time=pred_time[:,0:inputs.shape[-1]]
        assert pred_time.shape==inputs.shape, "bad shapes"
        return pred_time

            

class CropAddBlock(nn.Module):

    def forward(self,down_layer, x,  **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        #print(x1_shape,x2_shape)
        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2


        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff),:]
        x = torch.add(down_layer_cropped, x)
        return x

class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x


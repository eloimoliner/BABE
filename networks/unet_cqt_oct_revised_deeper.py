import torch.nn as nn
import numpy as np
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

    def __init__(self, num_features, num_groups=32, eps=1e-7):
        super(BiasFreeGroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        #self.beta = torch.zeros(1,num_features,1,1)
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups ,-1,H,W)
        #mean = x.mean(-1, keepdim=True)
        #var = x.var(-1, keepdim=True)

        std=x.std((2,3,4), keepdim=True) #reduce over channels and time
        #var = x.var(-1, keepdim=True)

        ## normalize
        x = (x) / (std+self.eps)
        # normalize
        x = x.view(N,C,H,W)
        return x * self.gamma





class RFF_MLP_Block(nn.Module):
    """
        Encoder of the noise level embedding
        Consists of:
            -Random Fourier Feature embedding
            -MLP
    """
    def __init__(self, emb_dim=512, rff_dim=32):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, rff_dim]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(2*rff_dim, 128),
            nn.Linear(128, 256),
            nn.Linear(256, emb_dim),
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



        
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=True,
        num_dils = 6,
        bias=False,
        kernel_size=(5,3),
        emb_dim=512,
        proj_place='before' #using 'after' in the decoder out blocks
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.num_dils=num_dils
        self.proj_place=proj_place


        self.res_conv = nn.Conv2d(dim, dim_out, 1,  bias=bias) #linear projection
        self.proj = nn.Conv2d(dim, dim_out, 1,  bias=bias) #linear projection

        if self.proj_place=='before':
            N=dim_out
        else:
            N=dim

        self.H=nn.ModuleList()
        self.affine=nn.ModuleList()
        if self.use_norm:
            self.norm=nn.ModuleList()

        for i in range(self.num_dils):

            if self.use_norm:
                self.norm.append(BiasFreeGroupNorm(N,8))

            self.affine.append(nn.Linear(emb_dim, N))
            #self.H.append(Gated_residual_layer(dim_out, (5,3), (2**i,1), bias=bias)) #sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)
            self.H.append(nn.Conv2d(N,N,    
                                    kernel_size=kernel_size,
                                    dilation=(2**i,1),
                                    padding='same',
                                    padding_mode='zeros', bias=bias)) #freq convolution (dilated) 


    def forward(self, input_x, sigma):
        
        x=input_x

        if self.proj_place=='before':
            x=self.proj(x)

        for norm, affine, conv in zip(self.norm, self.affine, self.H):

            x0=x
            if self.use_norm:
                x=norm(x)
            gamma =affine(sigma)
            x=x*(gamma.unsqueeze(2).unsqueeze(3)+1) #no bias
            x=(x0+conv(F.gelu(x)))/(2**0.5)
        
        if not(self.proj_place=='before'):
            x=self.proj(x)

        return (x + self.res_conv(input_x))/(2**0.5)

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
            self.mode_resample=mode_resample
            if mode_resample=="T":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            elif mode_resample=="F":
                #kerel shouuld be the same
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            else:
                raise NotImplementedError("Only time upsampling is implemented")
                #TODO implement freq upsampling and downsampling
            self.pad = kernel_1d.shape[0] // 2 - 1
            self.register_buffer('kernel', kernel_1d)
    def forward(self, x):
        shapeorig=x.shape
        #x=x.view(x.shape[0],-1,x.shape[-1])
        x=x.view(-1,x.shape[-2],x.shape[-1]) #I have the feeling the reshape makes everything consume too much memory. There is no need to have the channel dimension different than 1. I leave it like this because otherwise it requires a contiguous() call, but I should check if the memory gain / speed, would be significant.
        if self.mode_resample=="F":
            x=x.permute(0,2,1)#call contiguous() here?

        #print("after view",x.shape)
        if self.down:
            x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        elif self.up:
            x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)

        #print("after pad",x.shape)

        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        #print("weight",weight.shape)
        indices = torch.arange(x.shape[1], device=x.device)
        #print("indices",indices.shape)
        #weight = self.kernel.to(x.device).unsqueeze(0).unsqueeze(0).expand(x.shape[1], x.shape[1], -1)
        #print("weight",weight.shape)
        weight[indices, indices] = self.kernel.to(weight)
        if self.down:
            x_out= F.conv1d(x, weight, stride=2)
        elif self.up:
            x_out =F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)

        if self.mode_resample=="F":
            x_out=x_out.permute(0,2,1).contiguous()
            return x_out.view(shapeorig[0],-1,x_out.shape[-2], shapeorig[-1])
        else:
            return x_out.view(shapeorig[0],-1,shapeorig[2], x_out.shape[-1])


class Unet_CQT_oct_deeper(nn.Module):
    """
        Main U-Net model based on the CQT
    """
    def __init__(self, args, device):
        """
        Args:
            args (dictionary): hydra dictionary
            device: torch device ("cuda" or "cpu")
        """
        super(Unet_CQT_oct_deeper, self).__init__()
        self.args=args
        self.depth=args.network.depth
        self.inner_depth=args.network.inner_depth
        self.depth=args.network.inner_depth+self.args.network.cqt.num_octs
        assert self.depth==args.network.depth, "The depth of the network should be the sum of the inner depth and the number of octaves" #make sure we are aware of the depth of the network

        self.emb_dim=args.network.emb_dim
        self.embedding = RFF_MLP_Block(emb_dim=args.network.emb_dim)
        self.use_norm=args.network.use_norm

        #fmax=self.args.exp.sample_rate/2
        #self.fmin=fmax/(2**self.args.cqt.numocts)
        self.fbins=int(self.args.network.cqt.bins_per_oct*self.args.network.cqt.num_octs) 
        self.device=device
        self.bins_per_oct=self.args.network.cqt.bins_per_oct
        self.num_octs=self.args.network.cqt.num_octs
        #self.CQTransform=CQT_nsgt(self.args.network.cqt.num_octs,self.args.network.cqt.bins_per_oct, "oct",  self.args.exp.sample_rate, self.args.exp.audio_len, device=self.device)
        if self.args.network.cqt.window=="kaiser":
            win=("kaiser",self.args.network.cqt.beta)
        else:
            win=self.args.network.cqt.window

        self.CQTransform=CQT_nsgt(self.args.network.cqt.num_octs, self.args.network.cqt.bins_per_oct, mode="oct",window=win,fs=self.args.exp.sample_rate, audio_len=self.args.exp.audio_len, dtype=torch.float32, device=self.device)


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
        self.inner_Ns=self.args.network.inner_Ns
        self.Ss= self.args.network.Ss
        self.inner_s=self.args.network.inner_s

        self.num_dils= self.args.network.num_dils #intuition: less dilations for the first layers and more for the deeper layers
        self.inner_num_dils=self.args.network.inner_num_dils
        
        self.init_conv= nn.Conv2d(Nin,self.Ns[0],(5,3), padding="same", padding_mode="zeros", bias=False)


        self.downsamplerT=UpDownResample(down=True, mode_resample="T")
        self.downsamplerF=UpDownResample(down=True, mode_resample="F")
        self.upsamplerT=UpDownResample(up=True, mode_resample="T")
        self.upsamplerF=UpDownResample(up=True, mode_resample="F")

        self.downs=nn.ModuleList([])
        self.middle=nn.ModuleList([])
        self.ups=nn.ModuleList([])
        
        for i in range(self.num_octs):
            if i==0:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i-1]
                dim_out=self.Ns[i]

            self.downs.append(
                               nn.ModuleList([
                                        ResnetBlock(Nin, dim_in, self.use_norm,num_dils=1, bias=False, kernel_size=1, emb_dim=self.emb_dim),
                                        nn.Conv2d(2, dim_out, (5,3), padding="same", padding_mode="zeros", bias=False),
                                        ResnetBlock(dim_in, dim_out, self.use_norm,num_dils=self.num_dils[i], bias=False , emb_dim=self.emb_dim)
                                        ]))

        for i in range(self.inner_depth):
            if i==0:
                dim_in=self.Ns[-1]
                dim_out=self.inner_Ns[0]
            else:
                dim_in=self.inner_Ns[i-1]
                dim_out=self.inner_Ns[i]

            self.downs.append(
                               nn.ModuleList([
                                        nn.Conv2d(2, dim_out, (5,3), padding="same", padding_mode="zeros", bias=False),
                                        ResnetBlock(dim_in, dim_out, self.use_norm,num_dils=self.inner_num_dils[i], bias=False, emb_dim=self.emb_dim )
                                        ]))

        if self.args.network.bottleneck_type=="res_dil_convs":
            self.middle.append(nn.ModuleList([
                            ResnetBlock(self.inner_Ns[-1], self.inner_Ns[-1], self.use_norm, num_dils=self.num_dils[-1], bias=False, emb_dim=self.emb_dim)
                            ]))
        else:
            raise NotImplementedError("bottleneck type not implemented")
                        

        #self.pyr_up_proj_first=nn.Conv2d(dim_out, 2, (5,3), padding="same", padding_mode="zeros", bias=False)
        for i in range(self.args.network.inner_depth-1,-1,-1):

            if i==0:
                dim_in=self.inner_Ns[i]*2
                dim_out=self.Ns[-1]
            else:
                dim_in=self.inner_Ns[i]*2
                dim_out=self.inner_Ns[i-1]

            self.ups.append(nn.ModuleList(
                                        [
                                        #nn.Conv2d(dim_out, 2, (5,3), padding="same", padding_mode="zeros", bias=False),
                                        ResnetBlock(dim_in, dim_out, use_norm=self.use_norm,num_dils= self.inner_num_dils[i],bias=False, emb_dim=self.emb_dim)
                                        ]))

        for i in range(self.num_octs-1,-1,-1):

            if i==0:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i-1]


            self.ups.append(nn.ModuleList(
                                        [
                                        ResnetBlock(dim_out, 2, use_norm=self.use_norm,num_dils= 1,bias=False, kernel_size=1, proj_place="after", emb_dim=self.emb_dim),
                                        ResnetBlock(dim_in, dim_out, use_norm=self.use_norm,num_dils= self.num_dils[i],bias=False, emb_dim=self.emb_dim)
                                        ]))



        self.cropconcat = CropConcatBlock()




    def forward(self, inputs, sigma):
        """
        Args: 
            inputs (Tensor):  Input signal in time-domsin, shape (B,T)
            sigma (Tensor): noise levels,  shape (B,1)
        Returns:
            pred (Tensor): predicted signal in time-domain, shape (B,T)
        """
        #apply RFF embedding+MLP of the noise level
        sigma = self.embedding(sigma)

        
        #apply CQT to the inputs
        X_list =self.CQTransform.fwd(inputs.unsqueeze(1))
        X_list_out=X_list

        hs=[]
        for i,modules in enumerate(self.downs):
            if i <=(self.num_octs-1):
                C=X_list[-1-i]#get the corresponding CQT octave
                C=C.squeeze(1)
                C=torch.view_as_real(C)
                C=C.permute(0,3,1,2).contiguous() # call contiguous() here?
                if self.use_fencoding:
                    #Cfreq=self.freq_encoding(C)
                    C2=self.freq_encodings[i](C) #B, C + Nfreq*2, F,T
                else:
                    C2=C
                    
                init_block, pyr_down_proj, ResBlock=modules
                C2=init_block(C2,sigma)
            else:
                pyr_down_proj, ResBlock=modules
            
            if i==0:
                X=C2 #starting the main signal path
                pyr=self.downsamplerT(C) #starting the auxiliary path
            elif i<(self.num_octs-1):
                pyr=torch.cat((self.downsamplerT(C),self.downsamplerT(pyr)),dim=2) #updating the auxiliary path
                X=torch.cat((C2,X),dim=2) #updating the main signal path with the new octave
            elif i==(self.num_octs-1):# last layer
                pyr=torch.cat((self.downsamplerF(C),self.downsamplerF(pyr)),dim=2) #updating the auxiliary path
                X=torch.cat((C2,X),dim=2) #updating the main signal path with the new octave
            elif i>(self.num_octs-1) and i<(len(self.downs)-1):# last layer
                pyr=self.downsamplerF(pyr) #updating the auxiliary path
                X=X #updating the main signal path (no new octave)
            else: #last layer
                pass
                #pyr=pyr
                #X=X

            X=ResBlock(X, sigma)
            hs.append(X)

            #downsample the main signal path
            #we do not need to downsample in the inner layer
            if i<(self.num_octs-1): 
                X=self.downsamplerT(X)
                #apply the residual connection
                #X=(X+pyr_down_proj(pyr))/(2**0.5) #I'll my need to put that inside a combiner block??

            elif i>=(self.num_octs-1) and i<(len(self.downs)-1):
                X=self.downsamplerF(X)
                #apply the residual connection
                #X=(X+pyr_down_proj(pyr))/(2**0.5) #I'll my need to put that inside a combiner block??
            else: #last layer
                #no downsampling in the last layer
                pass

            #apply the residual connection
            X=(X+pyr_down_proj(pyr))/(2**0.5) #I'll my need to put that inside a combiner block??
            #print("encoder ", i, X.shape, X.mean().item(), X.std().item())
                
        #middle layers
        if self.args.network.bottleneck_type=="res_dil_convs":
            (ResBlock,)=self.middle[0]
            X=ResBlock(X, sigma)   

        #pyr=self.pyr_up_proj_first(X) Not usinf the pyr for now, maybe try it later?
        for i,modules in enumerate(self.ups):
            #print("decoder ", i, X.shape, X.mean().item(), X.std().item())
            j=len(self.ups) -i-1
            if j<=(self.num_octs-1):
                outBlock, ResBlock=modules
            else:
                (ResBlock,)=modules

            skip=hs.pop()
            X=torch.cat((X,skip),dim=1)
            X=ResBlock(X, sigma)

            if j<=(self.num_octs-1):
                X_out, X= X[:,:,0:self.bins_per_oct,:], X[:,:,self.bins_per_oct::,:]
                #pyr_out, pyr= pyr[:,:,0:self.bins_per_oct,:], pyr[:,:,self.bins_per_oct::,:]

                #X_out=(pyr_up_proj(X_out)+pyr_out)/(2**0.5)
                X_out=outBlock(X_out,sigma)

                X_out=X_out.permute(0,2,3,1).contiguous() #call contiguous() here?
                X_out=torch.view_as_complex(X_out)

                #save output
                X_list_out[i-self.inner_depth]=X_out.unsqueeze(1)
            elif j>(self.num_octs-1):
                pass

            if j>0 and j<=(self.num_octs-1):
                #pyr=self.upsampler(pyr) #call contiguous() here?
                X=self.upsamplerT(X) #call contiguous() here?
            elif j>(self.num_octs-1):
                X=self.upsamplerF(X)

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


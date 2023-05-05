import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math as m
import torch
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

from cqt_nsgt_pytorch import CQT_nsgt
import torchaudio
import einops

"""
As similar as possible to the original CQTdiff architecture, but using the octave-base representation of the CQT
This should be more memory efficient, and also more efficient in terms of computation, specially when using higher sampling rates.
I am expecting similar performance to the original CQTdiff architecture, but faster. 
Perhaps the fact that I am using powers of 2 for the time sizes is critical for transient reconstruction. I should thest CQT matrix model with powers of 2, this requires modifying the CQT_nsgt_pytorch.py file.
"""
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel=(1,1), bias=False, dilation=1,
        init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel[0]*kernel[1], fan_out=out_channels*kernel[0]*kernel[1])
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel[0], kernel[1]], **init_kwargs) * init_weight) 
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        #f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 1e-4, channel_last=True):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

class BiasFreeLayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-7):
        super(BiasFreeLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,1,num_features))
        #self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        #self.beta = torch.zeros(1,num_features,1,1)
        self.eps = eps

    def forward(self, x):
        N, T, C = x.size()
        #x = x.view(N, self.num_groups ,-1,H,W)
        #x=einops.rearrange(x, 'n t c -> n (t c)')
        #mean = x.mean(-1, keepdim=True)
        #var = x.var(-1, keepdim=True)

        std=x.std(-1, keepdim=True) #reduce over channels and time
        #var = x.var(-1, keepdim=True)

        ## normalize
        x = (x) / (std+self.eps)
        # normalize
        #x=einops.rearrange(x, 'n (t c) -> n t c', t=T)
        #x = x.view(N,C,H,W)
        return x * self.gamma

class BiasFreeGroupNorm(nn.Module):

    def __init__(self, num_features, num_groups=32, eps=1e-7):
        super(BiasFreeGroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        #self.beta = torch.zeros(1,num_features,1,1)
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, F, T = x.size()
        #x = x.view(N, self.num_groups ,-1,H,W)
        gc=C//self.num_groups
        x=einops.rearrange(x, 'n (g gc) f t -> n g (gc f t)', g=self.num_groups, gc=gc)
        #mean = x.mean(-1, keepdim=True)
        #var = x.var(-1, keepdim=True)

        std=x.std(-1, keepdim=True) #reduce over channels and time
        #var = x.var(-1, keepdim=True)

        ## normalize
        x = (x) / (std+self.eps)
        # normalize
        x=einops.rearrange(x, 'n g (gc f t) -> n (g gc) f t', g=self.num_groups, gc=gc, f=F, t=T)
        #x = x.view(N,C,H,W)
        return x * self.gamma



def FeedForward(dim: int, multiplier: int, init=None, init_zero=None) -> nn.Module:
    mid_features = dim * multiplier
    return nn.Sequential(
        BiasFreeLayerNorm(dim),
        Linear(in_features=dim, out_features=mid_features, bias=False, **init),
        nn.GELU(),
        Linear(in_features=mid_features, out_features=dim, bias=False, **init),
        LayerScale(dim, 1e-4)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.1,
        init=None,
        init_zero=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads


        self.norm = BiasFreeLayerNorm(dim)

        self.layerscale=LayerScale(dim, 1e-4)




        self.to_q = Linear(dim, inner_dim, bias = False, **init)
        self.to_kv = Linear(dim, dim_head * 2, bias = False, **init)
        self.to_out = Linear(inner_dim, dim, bias = False, **init_zero)

    def forward(
        self,
        x,
    ):

        kv_input = x

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)


        q = einops.rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        #reconsider if I want relative positional encoding
        #attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)

        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.layerscale(self.to_out(out))


class RFF_MLP_Block(nn.Module):
    """
        Encoder of the noise level embedding
        Consists of:
            -Random Fourier Feature embedding
            -MLP
    """
    def __init__(self, emb_dim=512, rff_dim=32, init=None):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, rff_dim]), requires_grad=False)
        self.MLP = nn.ModuleList([
            Linear(2*rff_dim, 128, **init),
            Linear(128, 256, **init),
            Linear(256, emb_dim, **init),
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
        proj_place='before', #using 'after' in the decoder out blocks
        init=None,
        init_zero=None,
        use_attention=False,
        attention_dict=None
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.num_dils=num_dils
        self.proj_place=proj_place

        if self.proj_place=='before':
            #dim_out is the block dimension
            N=dim_out
        else:
            #dim in is the block dimension
            N=dim

        #self.res_conv = Conv2d(dim, dim_out, bias=bias, **init) #linear projection
        #self.proj_in = Conv2d(dim, N,   bias=bias, **init) #linear projection
        #self.proj_out = Conv2d(N, dim_out,   bias=bias, **init_zero) #linear projection

        self.res_conv = Conv2d(dim, dim_out, bias=bias, **init) if dim!= dim_out else nn.Identity() #linear projection
        self.proj_in = Conv2d(dim, N,   bias=bias, **init) if dim!=N else nn.Identity()#linear projection
        self.proj_out = Conv2d(N, dim_out,   bias=bias, **init_zero) if N!=dim_out else nn.Identity() #linear projection


        self.H=nn.ModuleList()
        self.affine=nn.ModuleList()
        if self.use_norm:
            self.norm=nn.ModuleList()

        for i in range(self.num_dils):

            if self.use_norm:
                self.norm.append(BiasFreeGroupNorm(N,8))

            self.affine.append(Linear(emb_dim, N, **init))
            #self.H.append(Gated_residual_layer(dim_out, (5,3), (2**i,1), bias=bias)) #sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)
            self.H.append(Conv2d(N,N,    
                                    kernel=kernel_size,
                                    dilation=(2**i,1),
                                    bias=bias, **init)) #freq convolution (dilated) 


    def forward(self, input_x, sigma):
        
        x=input_x

        x=self.proj_in(x)

        for norm, affine, conv in zip(self.norm, self.affine, self.H):

            x0=x
            if self.use_norm:
                x=norm(x)
            gamma =affine(sigma)
            x=x*(gamma.unsqueeze(2).unsqueeze(3)+1) #no bias
            x=(x0+conv(F.gelu(x)))/(2**0.5)
        
        x=self.proj_out(x)

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


class Unet_CQT_oct_with_resdec(nn.Module):
    """
        Main U-Net model based on the CQT
    """
    def __init__(self, args, device):
        """
        Args:
            args (dictionary): hydra dictionary
            device: torch device ("cuda" or "cpu")
        """
        super(Unet_CQT_oct_with_resdec, self).__init__()
        self.args=args
        self.depth=args.network.depth
        self.inner_depth=args.network.inner_depth
        self.depth=args.network.inner_depth+self.args.network.cqt.num_octs
        assert self.depth==args.network.depth, "The depth of the network should be the sum of the inner depth and the number of octaves" #make sure we are aware of the depth of the network

        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3)) #same as ADM, according to edm implementation
        init_zero = dict(init_mode='kaiming_uniform', init_weight=1e-5) #I think it is safer to initialize the last layer with a small weight, rather than zero. Breaking symmetry and all that. 



        self.emb_dim=args.network.emb_dim
        self.embedding = RFF_MLP_Block(emb_dim=args.network.emb_dim, init=init)
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
                                        ResnetBlock(Nin, dim_in, self.use_norm,num_dils=1, bias=False, kernel_size=(1,1), emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                                        Conv2d(2, dim_out, kernel=(5,3), bias=False, **init),
                                        ResnetBlock(dim_in, dim_out, self.use_norm,num_dils=self.num_dils[i], bias=False , emb_dim=self.emb_dim, init=init, init_zero=init_zero)
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
                                        Conv2d(2, dim_out, kernel=(5,3), bias=False, **init),
                                        ResnetBlock(dim_in, dim_out, self.use_norm,num_dils=self.inner_num_dils[i], bias=False, emb_dim=self.emb_dim, init=init, init_zero=init_zero)
                                        ]))

        if self.args.network.bottleneck_type=="res_dil_convs":
            self.middle.append(nn.ModuleList([
                            ResnetBlock(self.inner_Ns[-1], 2, use_norm=self.use_norm,num_dils= 1,bias=False, kernel_size=(1,1), proj_place="after", emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                            ResnetBlock(self.inner_Ns[-1], self.inner_Ns[-1], self.use_norm, num_dils=self.num_dils[-1], bias=False, emb_dim=self.emb_dim, init=init, init_zero=init_zero),
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

            #print("decoder",i,dim_in, dim_out)
            self.ups.append(nn.ModuleList(
                                        [
                                        ResnetBlock(dim_out, 2, use_norm=self.use_norm,num_dils= 1,bias=False, kernel_size=(1,1), proj_place="after", emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                                        ResnetBlock(dim_in, dim_out, use_norm=self.use_norm,num_dils= self.inner_num_dils[i],bias=False, emb_dim=self.emb_dim, init=init, init_zero=init_zero),
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
                                        ResnetBlock(dim_out, 2, use_norm=self.use_norm,num_dils= 1,bias=False, kernel_size=(1,1), proj_place="after", emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                                        ResnetBlock(dim_in, dim_out, use_norm=self.use_norm,num_dils= self.num_dils[i],bias=False, emb_dim=self.emb_dim, init=init, init_zero=init_zero)
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
            OutBlock, ResBlock =self.middle[0]
            X=ResBlock(X, sigma)   
            Xout=OutBlock(X,sigma)


        for i,modules in enumerate(self.ups):
            j=len(self.ups) -i-1

            OutBlock,  ResBlock=modules

            skip=hs.pop()
            X=torch.cat((X,skip),dim=1)
            X=ResBlock(X, sigma)
            
            Xout=(Xout+OutBlock(X,sigma))/(2**0.5)


            if j<=(self.num_octs-1):
                X= X[:,:,self.bins_per_oct::,:]
                Out, Xout= Xout[:,:,0:self.bins_per_oct,:], Xout[:,:,self.bins_per_oct::,:]
                #pyr_out, pyr= pyr[:,:,0:self.bins_per_oct,:], pyr[:,:,self.bins_per_oct::,:]
                #X_out=(pyr_up_proj(X_out)+pyr_out)/(2**0.5)

                Out=Out.permute(0,2,3,1).contiguous() #call contiguous() here?
                Out=torch.view_as_complex(Out)

                #save output
                X_list_out[i-self.inner_depth]=Out.unsqueeze(1)

            elif j>(self.num_octs-1):
                pass

            if j>0 and j<=(self.num_octs-1):
                #pyr=self.upsampler(pyr) #call contiguous() here?
                X=self.upsamplerT(X) #call contiguous() here?
                Xout=self.upsamplerT(Xout) #call contiguous() here?
            elif j>(self.num_octs-1):
                X=self.upsamplerF(X)
                Xout=self.upsamplerF(Xout)

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


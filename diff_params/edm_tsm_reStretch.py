import torch
import numpy as np

import utils.training_utils as utils

import utils.tsm_utils as tsm_utils

class EDM():
    """
        Definition of most of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022)
    """

    def __init__(self, args):
        """
        Args:
            args (dictionary): hydra arguments
            sigma_data (float): 
        """
        self.args=args
        self.sigma_min = args.diff_params.sigma_min
        self.sigma_max =args.diff_params.sigma_max
        self.P_mean=args.diff_params.P_mean
        self.P_std=args.diff_params.P_std
        self.ro=args.diff_params.ro
        self.ro_train=args.diff_params.ro_train
        self.sigma_data=args.diff_params.sigma_data #depends on the training data!! precalculated variance of the dataset
        #parameters stochastic sampling
        self.Schurn=args.diff_params.Schurn
        self.Stmin=args.diff_params.Stmin
        self.Stmax=args.diff_params.Stmax
        self.Snoise=args.diff_params.Snoise

        #perceptual filter
        if self.args.diff_params.aweighting.use_aweighting:
            self.AW=utils.FIRFilter(filter_type="aw", fs=args.exp.sample_rate, ntaps=self.args.diff_params.aweighting.ntaps)

       

    def get_gamma(self, t): 
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N=t.shape[0]
        gamma=torch.zeros(t.shape).to(t.device)
        
        #If desired, only apply stochasticity between a certain range of noises Stmin is 0 by default and Stmax is a huge number by default. (Unless these parameters are specified, this does nothing)
        indexes=torch.logical_and(t>self.Stmin , t<self.Stmax)
         
        #We use Schurn=5 as the default in our experiments
        gamma[indexes]=gamma[indexes]+torch.min(torch.Tensor([self.Schurn/N, 2**(1/2) -1]))
        
        return gamma

    def create_schedule(self,nb_steps):
        """
        Define the schedule of timesteps
        Args:
           nb_steps (int): Number of discretized steps
        """
        i=torch.arange(0,nb_steps+1)
        t=(self.sigma_max**(1/self.ro) +i/(nb_steps-1) *(self.sigma_min**(1/self.ro) - self.sigma_max**(1/self.ro)))**self.ro
        t[-1]=0
        return t

    def create_schedule_from_initial_t(self,initial_t,nb_steps):
        """
        Define the schedule of timesteps
        Args:
           nb_steps (int): Number of discretized steps
        """
        i=torch.arange(0,nb_steps+1)
        t=(initial_t**(1/self.ro) +i/(nb_steps-1) *(self.sigma_min**(1/self.ro) - initial_t**(1/self.ro)))**self.ro
        t[-1]=0
        return t


    def sample_ptrain(self,N):
        """
        For training, getting t as a normal distribution, folowing Karras et al. 
        I'm not using this
        Args:
            N (int): batch size
        """
        lnsigma=np.random.randn(N)*self.P_std +self.P_mean
        return np.clip(np.exp(lnsigma),self.sigma_min, self.sigma_max) #not sure if clipping here is necessary, but makes sense to me
    
    def sample_ptrain_safe(self,N):
        """
        For training, getting  t according to the same criteria as sampling
        Args:
            N (int): batch size
        """
        a=torch.rand(N)
        t=(self.sigma_max**(1/self.ro_train) +a *(self.sigma_min**(1/self.ro_train) - self.sigma_max**(1/self.ro_train)))**self.ro_train
        return t

    def sample_prior(self,shape,sigma):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
            sigma (float): noise level of the noise
        """
        n=torch.randn(shape, device=sigma.device)*sigma
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 *(sigma**2+self.sigma_data**2)**-1

    def cout(self,sigma ):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma*self.sigma_data* (self.sigma_data**2+sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2+sigma**2)**(-0.5)

    def cnoise(self,sigma ):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1/4)*torch.log(sigma)

    def lambda_w(self,sigma):
        return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)

    def denoiser_cfg_spectrogram(self, xNn , net, sigma, xSn=None, xTn=None, CLAP=None, Nspec=None, cfg_args=None):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
            xSn (Tensor): shape: (B,T) sines signal (with noise)
            xTn (Tensor): shape: (B,T) transient signal (with noise)
            CLAP (Tensor): shape: (B,D) CLAP latent features
            cfg (float): cfg parameter, classifier-free guidance (default 1.0) 1.0 means no guidance, shoud be superior to 1.0
        """
        if len(sigma.shape)==1:
            sigma=sigma.unsqueeze(-1)
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)


        try:
            if self.args.diff_params.noise_ST=="None":
                #just zeroes
                cinS=1
                cinT=1
        
            if self.args.diff_params.noise_ST=="same_as_x":
                cinS=cin
                cinT=cin
        except:
            #if that variable is not defined, just use the same noise for all (same_as_x)
            cinS=cin
            cinT=cin

        if cfg_args.mode=="alltogether":
            raise NotImplementedError
            #batch two copies of XNn
            xNn_in = torch.cat([xNn] * 2)
            cnoise_in=torch.cat([cnoise]*2)
          
            #same xSn, but zero the second one
            xSn_in=torch.cat([xSn,xSn*0],dim=0)
            xTn_in=torch.cat([xTn,xTn*0],dim=0)
            CLAP_in=torch.cat([CLAP,CLAP*0],dim=0)
          
            #print("xN_in", xNn_in.shape, "cnoise_in", cnoise_in.shape, "xSn_in", xSn_in.shape, "xTn_in", xTn_in.shape, "CLAP_in", CLAP_in.shape)
            #e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            assert cfg_args.CLAP == cfg_args.ST, "CLAP and ST must be the same, now they are {} and {}".format(cfg_args.CLAP, cfg_args.ST)
            cfg=cfg_args.CLAP
          
            out_cond, out_uncond=net(cin*xNn_in, cnoise_in, cinS*xSn_in, cinT*xTn_in, CLAP_in).chunk(2, dim=0)
            #return cskip * xNn +cout*((1+cfg)*out_cond -cfg*out_uncond)  #this will crash because of broadcasting problems, debug later!
            return cskip * xNn +cout*(out_uncond+cfg*(out_cond-out_uncond))
        elif cfg_args.mode=="separated":
            #batch two copies of XNn
            xNn_in = torch.cat([xNn] * 5)
            cnoise_in=torch.cat([cnoise]*5)
          
            #same xSn, but zero the second one
            xSn_in=torch.cat([xSn, xSn*0, xSn*0, xSn*0, xSn*0],dim=0)
            xTn_in=torch.cat([xTn*0, xTn,xTn*0,xTn*0, xTn*0],dim=0)
            CLAP_in=torch.cat([CLAP*0, CLAP*0, CLAP, CLAP*0, CLAP*0],dim=0)
            Nspec_in=torch.cat([Nspec*0, Nspec*0, Nspec*0, Nspec, Nspec*0],dim=0)
          
            #print("xN_in", xNn_in.shape, "cnoise_in", cnoise_in.shape, "xSn_in", xSn_in.shape, "xTn_in", xTn_in.shape, "CLAP_in", CLAP_in.shape)
            #e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            cfg_CLAP=cfg_args.CLAP
            cfg_S=cfg_args.S
            cfg_T=cfg_args.T
            cfg_Nspec=cfg_args.Nspec
          
            out_cond_S, out_cond_T, out_cond_CLAP, out_cond_Nspec,out_uncond=net(cin*xNn_in, cnoise_in, cinS*xSn_in, cinT*xTn_in, CLAP_in, Nspec_in).chunk(5, dim=0)
            #return cskip * xNn +cout*((1+cfg_CLAP)*out_cond_CLAP -cfg*out_uncond)  #this will crash because of broadcasting problems, debug later!
            return cskip * xNn +cout*(out_uncond+cfg_CLAP*(out_cond_CLAP-out_uncond)+cfg_S*(out_cond_S-out_uncond)+cfg_T*(out_cond_T-out_uncond) +cfg_Nspec*(out_cond_Nspec- out_uncond))  #this will crash because of broadcasting problems, debug later!
        else:
            raise NotImplementedError

        
    def denoiser_cfg(self, xNn , net, sigma, xSn=None, xTn=None, CLAP=None,xN_restretched=None, cfg_args=None):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
            xSn (Tensor): shape: (B,T) sines signal (with noise)
            xTn (Tensor): shape: (B,T) transient signal (with noise)
            CLAP (Tensor): shape: (B,D) CLAP latent features
            cfg (float): cfg parameter, classifier-free guidance (default 1.0) 1.0 means no guidance, shoud be superior to 1.0
        """
        if len(sigma.shape)==1:
            sigma=sigma.unsqueeze(-1)
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)


        if cfg_args.mode=="alltogether":
            #batch two copies of XNn
            xNn_in = torch.cat([xNn] * 2)
            cnoise_in=torch.cat([cnoise]*2)
          
            #same xSn, but zero the second one
            xSn_in=torch.cat([xSn,xSn*0],dim=0)
            xTn_in=torch.cat([xTn,xTn*0],dim=0)
            CLAP_in=torch.cat([CLAP,CLAP*0],dim=0)
            xN_restretched_in=torch.cat([xN_restretched,xN_restretched*0],dim=0)
          
            #print("xN_in", xNn_in.shape, "cnoise_in", cnoise_in.shape, "xSn_in", xSn_in.shape, "xTn_in", xTn_in.shape, "CLAP_in", CLAP_in.shape)
            #e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            assert cfg_args.CLAP == cfg_args.ST, "CLAP and ST must be the same, now they are {} and {}".format(cfg_args.CLAP, cfg_args.ST)
            cfg=cfg_args.CLAP
          
            out_cond, out_uncond=net(cin*xNn_in, cnoise_in, cin*xSn_in, cin*xTn_in, CLAP_in).chunk(2, dim=0)
            #return cskip * xNn +cout*((1+cfg)*out_cond -cfg*out_uncond)  #this will crash because of broadcasting problems, debug later!
            return cskip * xNn +cout*(out_uncond+cfg*(out_cond-out_uncond))
        elif cfg_args.mode=="separated":
            #batch two copies of XNn
            xNn_in = torch.cat([xNn] * 5)
            cnoise_in=torch.cat([cnoise]*5)
          
            #same xSn, but zero the second one
            xSn_in=torch.cat([xSn, xSn*0, xSn*0, xSn*0, xSn*0],dim=0)
            xTn_in=torch.cat([xTn*0, xTn,xTn*0, xTn*0,xTn*0],dim=0)
            CLAP_in=torch.cat([CLAP*0, CLAP*0, CLAP, CLAP*0, CLAP*0],dim=0)
            xN_restretched_in=torch.cat([xN_restretched*0, xN_restretched*0, xN_restretched*0, xN_restretched, xN_restretched*0],dim=0)
          
            #print("xN_in", xNn_in.shape, "cnoise_in", cnoise_in.shape, "xSn_in", xSn_in.shape, "xTn_in", xTn_in.shape, "CLAP_in", CLAP_in.shape)
            #e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            cfg_CLAP=cfg_args.CLAP
            cfg_S=cfg_args.S
            cfg_T=cfg_args.T
            cfg_xN_restretched=cfg_args.xN_restretched
          
            out_cond_S, out_cond_T, out_cond_CLAP, out_cond_xN, out_uncond=net(cin*xNn_in, cnoise_in, cin*xSn_in, cin*xTn_in, CLAP_in, cin*xN_restretched_in).chunk(5, dim=0)
            #return cskip * xNn +cout*((1+cfg_CLAP)*out_cond_CLAP -cfg*out_uncond)  #this will crash because of broadcasting problems, debug later!
            return cskip * xNn +cout*(out_uncond+cfg_CLAP*(out_cond_CLAP-out_uncond)+cfg_S*(out_cond_S-out_uncond)+cfg_T*(out_cond_T-out_uncond)+cfg_xN_restretched*(out_cond_xN-out_uncond))  #this will crash because of broadcasting problems, debug later!
        else:
            raise NotImplementedError



    def denoiser(self, xNn , net, sigma, xSn=None, xTn=None, CLAP=None, xN_stretched=None):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
            xSn (Tensor): shape: (B,T) sines signal (with noise)
            xTn (Tensor): shape: (B,T) transient signal (with noise)
            CLAP (Tensor): shape: (B,D) CLAP latent features
        """
        if len(sigma.shape)==1:
            sigma=sigma.unsqueeze(-1)
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)


        return cskip * xNn +cout*net(cin*xNn, cnoise, cin*xSn, cin*xTn, CLAP, cin*xN_stretched)  #this will crash because of broadcasting problems, debug later!

    def prepare_train_preconditioning(self, x, xS, xT, sigma):
        #weight=self.lambda_w(sigma)
        #Is calling the denoiser here a good idea? Maybe it would be better to apply directly the preconditioning as in the paper, even though Karras et al seem to do it this way in their code
        noise=self.sample_prior(x.shape,sigma.unsqueeze(1))
        #noise=noise.unsqueeze(1)
        #noiseS=noiseS.unsqueeze(1)
        #noiseT=noiseT.unsqueeze(1)

        #print("noise shape", noise.shape, sigma.shape)

        cskip=self.cskip(sigma).unsqueeze(1)
        cout=self.cout(sigma).unsqueeze(1)
        cin=self.cin(sigma).unsqueeze(1)
        cnoise=self.cnoise(sigma)
        #print("cin", cin.shape, "cout", cout.shape, "cskip", cskip.shape, "cnoise", cnoise.shape)
        try:
            if self.args.diff_params.noise_ST=="None":
                #just zeroes
                noiseS=torch.zeros_like(xS)
                noiseT=torch.zeros_like(xT)
                cinS=cin
                cinT=cin
        
            if self.args.diff_params.noise_ST=="same_as_x":
                noiseS=self.sample_prior(xS.shape,sigma.unsqueeze(1))
                noiseT=self.sample_prior(xT.shape,sigma.unsqueeze(1))
                cinS=cin
                cinT=cin
        except:
            #if that variable is not defined, just use the same noise for all (same_as_x)
            noiseS=self.sample_prior(xS.shape,sigma.unsqueeze(1))
            noiseT=self.sample_prior(xT.shape,sigma.unsqueeze(1))
            cinS=cin
            cinT=cin

        target=(1/cout)*(x-cskip*(x+noise))

        #lets use the same noise for the SR for now...
        return cin*(x+noise), target, cinS*(xS+noiseS), cinT*(xT+noiseT), cnoise

    def CFG_dropout(self, x):
        """
        Apply dropout to the conditioners so that we can apply Classifier-free guidance
        """
        rate=self.args.diff_params.cfg_dropout_rate
        #randomly set some of the conditioners instances to zero
        B=x.shape[0]

        #the shape S shold be the same as x.shape, but with all the dimensions but the first one set to 1
        S=(B,)+tuple([1 for i in range(len(x.shape)-1)])
        #print(S, x.shape)

        x=torch.where(torch.rand(S, device=x.device)<rate, torch.zeros_like(x), x)
        #retrieve the shape that goes to rand from the shape of x


        return x

    def CFG_dropout_old(self, xS, xT, xN, CLAP_latent):
        """
        Apply dropout to the conditioners so that we can apply Classifier-free guidance
        """
        rate=self.args.diff_params.cfg_dropout_rate
        #randomly set some of the conditioners instances to zero
        B=xS.shape[0]
        xS=torch.where(torch.rand((B,1,1), device=xS.device)<rate, torch.zeros_like(xS), xS)
        xT=torch.where(torch.rand((B,1,1), device=xT.device)<rate, torch.zeros_like(xT), xT)
        xN=torch.where(torch.rand((B,1,1), device=xN.device)<rate, torch.zeros_like(xN), xN)
        CLAP_latent=torch.where(torch.rand((B,1), device=CLAP_latent.device)<rate, torch.zeros_like(CLAP_latent), CLAP_latent)
        #print to see if it works
        #print("xS dropout", xS.std(-1))
        #print("xT dropout", xT.std(-1))
        #print("CLAP dropout", CLAP_latent.std(-1))

        return xS, xT, xN, CLAP_latent



    def reStretch(self, x, restretching):
        """
        Compress and stretch x with the specified restretching factors
        """
        #compress
        return tsm_utils.reStretch_N(x, restretching)


        

    def loss_fn(self, net, xN, xS, xT, CLAP_latent=None ):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            xN (Tensor): shape: (B,T) Intermediate noisy latent to denoise (residuals)
            xS (Tensor): shape: (B,T) Intermediate noisy latent to denoise (sines)
            xT (Tensor): shape: (B,T) Intermediate noisy latent to denoise (transients)
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        xN_reStretched=tsm_utils.reStretch_N(xN,self.args.diff_params.restretching.total_factor)


        sigma=self.sample_ptrain_safe(xN.shape[0]).unsqueeze(-1).to(xN.device)

        #stack sines and transients

        input, target, xS, xT, cnoise= self.prepare_train_preconditioning(xN, xS,xT, sigma)
        #print(input.shape, target.shape, xS.shape, xT.shape, cnoise.shape)
        #TODO: add classifier-free guidance on both conditions!!!!
        #print(xS.shape, xT.shape, CLAP_latent.shape)
        #make sure that the input variable is not the same as xN, because that would cause problems with the gradient   

        xS=self.CFG_dropout(xS)
        xT=self.CFG_dropout(xT)
        CLAP_latent=self.CFG_dropout(CLAP_latent)
        xN_reStretched=self.CFG_dropout(xN_reStretched)



        #print("xNspec",xNspec)
        #print(xS.shape, xT.shape, CLAP_latent.shape)
        #try:
        #with a certain probability, train unconditioned
        if torch.rand((1,))<self.args.diff_params.cfg_unconditional_rate:
            estimate=net(input,cnoise, xS*0, xT*0, CLAP_latent*0, xN_reStretched*0)
        else:
            estimate=net(input,cnoise, xS, xT, CLAP_latent, xN_reStretched)
        
        #except Exception as e:
        #        print("Exception in loss_fn, trying agai")
        #        print(e)
        #        estimate=net(input,cnoise, xS, xT, CLAP_latent)
        
        error=(estimate-target)

        try:
            #this will only happen if the model is cqt-based, if it crashes it is normal
            if self.args.network.filter_out_cqt_DC_Nyq:
                error=net.CQTransform.apply_hpf_DC(error) #apply the DC correction to the error as we dont want to propagate the DC component of the error as the network is discarding it. It also applies for the nyquit frequency, but this is less critical.
        except:
            pass 

        #APPLY A-WEIGHTING
        if self.args.diff_params.aweighting.use_aweighting:
            error=self.AW(error)

        #here we have the chance to apply further emphasis to the error, as some kind of perceptual frequency weighting could be
        return error**2, sigma


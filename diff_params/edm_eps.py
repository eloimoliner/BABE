import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import utils.training_utils as utils


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

        #parameters from: https://github.com/yoyololicon/diffwave-sr/blob/main/ckpt/vctk_48k_udm/.hydra/config.yaml

        self.T=self.args.diff_params.T

        self.gamma1=torch.Tensor([self.args.diff_params.scheduler.gamma1])
        self.gamma0=torch.Tensor([self.args.diff_params.scheduler.gamma0])
        print("gamam0, 1",self.gamma0, self.gamma1)
       
        t = torch.linspace(0, 1, self.T + 1)
        self.gamma, t=self.LogSNRLinearScheduler(self.gamma1, self.gamma0, t)


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

    def LogSNRLinearScheduler(self, gamma1, gamma0, t):

        t = t.clamp(0, 1)
        gamma = gamma0 * (1 - t) + gamma1 * t

        return gamma, t
    def gamma_to_t(self, gamma):
        """
        Convert the parameter gamma to the parameter t
        Args:
            gamma (Tensor): shape: (N_steps, ) Tensor of gamma values
        """
        return (gamma - self.gamma0) / (self.gamma1 - self.gamma0)

    def t_to_gamma(self, t):
        """
        Convert the parameter t to the parameter gamma"""
        return self.gamma0 + t * (self.gamma1 - self.gamma0)

    def gamma_2_as(self, gamma):
        """
        Convert the parameter gamma to the parameter alpha and s
        Args:
            gamma (Tensor): shape: (N_steps, ) Tensor of gamma values
        """
        var = gamma.sigmoid()
        return (1 - var).sqrt(), var.sqrt()
    
    def t_2_as(self, t):
        """
        Convert the parameter t to the parameter alpha and s
        Args:
            t (Tensor): shape: (N_steps, ) Tensor of t values
        """
        gamma=self.t_to_gamma(t)
        return self.gamma_2_as(gamma)



    def gamma_to_sigma(self, gamma):
        """
        Convert the parameter gamma to the parameter sigma
        Args:
            gamma (Tensor): shape: (N_steps, ) Tensor of gamma values
        """
        return torch.sqrt(1/torch.exp(-gamma))

    def sigma_to_gamma(self, sigma):
        """
        Convert the parameter sigma to the parameter gamma
        Args:
            sigma (Tensor): shape: (N_steps, ) Tensor of sigma values
        """
        return torch.log(sigma**2)
    
    def sigma_to_t(self, sigma):
        gamma=self.sigma_to_gamma(sigma)
        print("gamma",gamma)
        return self.gamma_to_t(gamma)

    def gamma2logas(self,g):
        log_var = -F.softplus(-g)
        return 0.5 * (-g + log_var), log_var
       
    def reverse_process_ddim(self,z_1,  model):

        tt = torch.linspace(0, 1, self.T + 1)
        gamma, steps =self.LogSNRLinearScheduler(self.gamma1, self.gamma0, tt)
        #print(gamma, steps)
        Pm1 = -torch.expm1((gamma[1:] - gamma[:-1]) * 0.5)
        log_alpha, log_var = self.gamma2logas(gamma)
        #print("log_alpha", log_alpha, "log_var", log_var)
        alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
        #print("alpha_st", alpha_st)
        std = log_var.mul(0.5).exp()

        T = gamma.numel() - 1
        z_t = z_1
        for t in tqdm(range(T, 0, -1)):
            #print("z_t std", z_t.std(-1))
            s = t - 1
            #print("steps",steps[t:t+1],"gamma", gamma[t], "alpha_st", alpha_st[s], "std",std[s], "pm1", Pm1[s])
            #print(z_t.shape, steps[t:t+1].shape)
            noise_hat = model(z_t, steps[t:t+1])
            noise_hat = noise_hat.float()
            z_t.mul_(alpha_st[s]).add_(std[s] * Pm1[s] * noise_hat)

        return z_t


    def get_gamma(self, t): 
        """
        Get the parameter gamma that defines the stochasticity of the sampler, it is not the same as the parameter gamma in the scheduler
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
        n=torch.randn(shape).to(sigma.device)*sigma
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
        
    def denoiser(self, xn , net, sigma):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        if len(sigma.shape)==1:
            sigma=sigma.unsqueeze(-1)
        #cskip=self.cskip(sigma)
        #cout=self.cout(sigma)
        #cin=self.cin(sigma)
        #cnoise=self.cnoise(sigma)
        self.gamma0=self.gamma0.to(sigma.device)
        self.gamma1=self.gamma1.to(sigma.device)


        #print(sigma.device)
        #t=self.sigma_to_t(sigma)

        gamma=self.sigma_to_gamma(sigma)
        #print("gamma", gamma)
        t=self.gamma_to_t(gamma)
        a, s=self.gamma_2_as(gamma)
        #print(a.shape, s.shape, sigma.shape, t.shape)
        #print("sigma", sigma, "t", t)
        #print("a", a,"s", s)

        z_t=a*xn #this is equivalent to cin
        #print("z_t std",z_t.std(-1))

        t=t.expand(z_t.shape[0],1).squeeze(-1) 
        #print("before net", z_t.shape, t.shape)
        eps_hat=net(z_t, t)
        eps_hat=eps_hat

        x0_hat=(-s*eps_hat+z_t)/a #equvalent to cout and cskip

        return x0_hat#this will crash because of broadcasting problems, debug later!

    def prepare_train_preconditioning(self, x, sigma):
        #weight=self.lambda_w(sigma)
        #Is calling the denoiser here a good idea? Maybe it would be better to apply directly the preconditioning as in the paper, even though Karras et al seem to do it this way in their code
        print(x.shape)
        noise=self.sample_prior(x.shape,sigma)

        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)

        target=(1/cout)*(x-cskip*(x+noise))

        return cin*(x+noise), target, cnoise


    def loss_fn(self, net, x):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma=self.sample_ptrain_safe(x.shape[0]).unsqueeze(-1).to(x.device)

        input, target, cnoise= self.prepare_train_preconditioning(x, sigma)
        estimate=net(input,cnoise)
        
        error=(estimate-target)

        try:
            #this will only happen if the model is cqt-based, if it crashes it is normal
            if self.args.net.use_cqt_DC_correction:
                error=net.CQTransform.apply_hpf_DC(error) #apply the DC correction to the error as we dont want to propagate the DC component of the error as the network is discarding it. It also applies for the nyquit frequency, but this is less critical.
        except:
            pass 

        #APPLY A-WEIGHTING
        if self.args.diff_params.aweighting.use_aweighting:
            error=self.AW(error)

        #here we have the chance to apply further emphasis to the error, as some kind of perceptual frequency weighting could be
        return error**2, sigma


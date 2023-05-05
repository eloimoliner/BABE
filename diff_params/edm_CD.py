import torch
import numpy as np

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

        self.boundaries=self.create_schedule(self.args.diff_params.CD.boundaries.T)
        #print(self.boundaries)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.boundaries=self.boundaries[:-2]
        self.boundaries=torch.flip(self.boundaries,(0,)).to(self.device)
        #print("real T", len(self.boundaries))
        #print(self.boundaries)
       

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
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)

        return cskip * xn +cout*net(cin*xn, cnoise)  #this will crash because of broadcasting problems, debug later!

    def prepare_train_preconditioning(self, x, sigma):
        #weight=self.lambda_w(sigma)
        #Is calling the denoiser here a good idea? Maybe it would be better to apply directly the preconditioning as in the paper, even though Karras et al seem to do it this way in their code
        #print(x.shape)
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
        #print("inputs to net", input.shape, cnoise.shape)
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

    def ode_update(self, x, sigma_0, sigma_1, net_teacher):
        x_0_hat=self.denoiser(x, net_teacher, sigma_1)
        score=(x_0_hat-x)/sigma_1**2
        return x-(sigma_0-sigma_1)*sigma_1*score

    def loss_fn_CD(self, net,ema, net_teacher, x):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """

        #sigma=self.sample_ptrain_safe(x.shape[0]).unsqueeze(-1).to(x.device)

        #sample sigma using self.boundaries
        i = torch.randint(0, len(self.boundaries) - 1, (x.shape[0], 1), device=x.device)
        sigma_0=self.boundaries[i]
        sigma_1=self.boundaries[i+1]
        #print(i, sigma_0, sigma_1)

        
         
        input_0, _ , cnoise_0= self.prepare_train_preconditioning(x, sigma_0)
        input_1, _ , cnoise_1= self.prepare_train_preconditioning(x, sigma_1)


        #print("inputs to net", input.shape, cnoise.shape)
        #student forward pass
        noise=self.sample_prior(x.shape,sigma_1)
        
        estimate=self.denoiser(x+noise, net, sigma_1 )
        
        with torch.no_grad():
            x_hat_0=self.ode_update(input_1, sigma_0,sigma_1, net_teacher)
            target=self.denoiser(x_hat_0, ema, sigma_0)
        
        #print(estimate.std(-1), target.std(-1))
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
        return error**2, sigma_0

    def CD_sample_longer(self, N, L, net, device, numsteps=1,pre_schedule=None):
        """
           N: batch size
           L: length
           device
        """
        prior=torch.randn((N,L), device=device)*self.boundaries[-1]
        schedule=[]
        schedule+=[self.boundaries[-1]]
        if numsteps>1:
            a=int(len(self.boundaries)//numsteps)
            for i in range(numsteps-1,0,-1):
                if pre_schedule is not None:
                    print(i)
                    pre_sch=pre_schedule[i-1]
                    j=int(len(self.boundaries)*pre_sch)
                else:
                    j=(i)*a
                schedule+=[self.boundaries[j]]

        print(schedule)
        mask=torch.zeros_like(prior)
        samples_ov=int(self.args.tester.CD.outpainting.overlap*self.args.exp.sample_rate)
        hop=L-samples_ov

        mask[...,0:samples_ov]=1
                
        
        not_finished=True
        seconds_processed=0
        output=torch.zeros((N,int(self.args.tester.CD.outpainting.length*self.args.exp.sample_rate)), device=device)
        
        pointer=0
        #first an unconditional generation
        with torch.no_grad(): 
             prior=torch.randn((N,L), device=device)*self.boundaries[-1]
             x_0_hat=self.denoiser(prior,net, schedule[0])
             if numsteps>1:
                for i in range(numsteps-1):
                    #add noise
                    print("predictor corrector step at", schedule[i+1])
                    x=x_0_hat+torch.randn((N,L), device=device)*schedule[i+1]
                    x_0_hat=self.denoiser(x, net, schedule[i+1])

        seconds_processed+=hop/self.args.exp.sample_rate
        output[...,pointer:(pointer+L)]=x_0_hat
        pointer+=hop

        print(seconds_processed, self.args.tester.CD.outpainting.length)
        while (seconds_processed+L/self.args.exp.sample_rate)<self.args.tester.CD.outpainting.length:
              print(seconds_processed, self.args.tester.CD.outpainting.length)
              x_pred=output[..., pointer:(pointer+L)]
              with torch.no_grad(): 
                  prior=torch.randn((N,L), device=device)*self.boundaries[-1]
                  x_0_hat=self.denoiser(prior,net, schedule[0])
                  if numsteps>1:
                       for i in range(numsteps-1):
                           #add noise
                           print("predictor corrector step at", schedule[i+1])
                           x_masked=mask*x_pred+(1-mask)*x_0_hat
                           x=x_masked+torch.randn((N,L), device=device)*schedule[i+1]
                           x_0_hat=self.denoiser(x, net, schedule[i+1])

              #now autoregressive
              seconds_processed+=hop/self.args.exp.sample_rate

              output[..., pointer:(pointer+L)]=x_0_hat

              pointer+=hop
        
        #last chunk
        if seconds_processed+samples_ov/self.args.exp.sample_rate < self.args.tester.CD.outpainting.length:
              x_pred=output[..., pointer::] 
              x_pred=torch.cat((x_pred, torch.zeros(x_pred.shape[0], L-x_pred.shape[-1])), -1)

              with torch.no_grad(): 
                  prior=torch.randn((N,L), device=device)*self.boundaries[-1]
                  x_0_hat=self.denoiser(prior,net, schedule[0])
                  if numsteps>1:
                       for i in range(numsteps-1):
                           #add noise
                           print("predictor corrector step at", schedule[i+1])
                           x_masked=mask*x_pred+(1-mask)*x_0_hat
                           x=x_masked+torch.randn((N,L), device=device)*schedule[i+1]
                           x_0_hat=self.denoiser(x, net, schedule[i+1])

              #now autoregressive
              seconds_processed+=hop/self.args.exp.sample_rate

              last_samples=output[...,pointer::]
              output[..., pointer::]=x_0_hat[...,0:last_samples.shape[-1]]
              
        
        return output

    def CD_sample(self, N, L, net, device, numsteps=1):
        """
           N: batch size
           L: length
           device
        """
        prior=torch.randn((N,L), device=device)*self.boundaries[-1]
        schedule=[]
        schedule+=[self.boundaries[-1]]
        if numsteps>1:
            a=int(len(self.boundaries)//numsteps)
            for i in range(numsteps-1,0,-1):
                schedule+=[self.boundaries[(i)*a]]

        with torch.no_grad(): 
             x_0_hat=self.denoiser(prior,net, schedule[0])
             if numsteps>1:
                for i in range(numsteps-1):
                    #add noise
                    print(schedule, i)
                    print("predictor corrector step at", schedule[i+1])
                    x=x_0_hat+torch.randn((N,L), device=device)*schedule[i+1]
                    x_0_hat=self.denoiser(x, net, schedule[i+1])
        
        return x_0_hat


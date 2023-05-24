from tqdm import tqdm
import torch
import torchaudio
#import scipy.signal
import copy
#import numpy as np
#import utils.filter_generation_utils as f_utils

import utils.blind_bwe_utils as blind_bwe_utils


class BlindSampler():

    def __init__(self, model,  diff_params, args, rid=False):

        self.model = model

        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args
        if not(self.args.tester.diff_params.same_as_training):
            self.update_diff_params()


        self.order=self.args.tester.order

        self.xi=self.args.tester.posterior_sampling.xi
        #hyperparameter for the reconstruction guidance
        self.data_consistency=self.args.tester.posterior_sampling.data_consistency #use reconstruction gudance without replacement
        self.nb_steps=self.args.tester.T

        #self.treshold_on_grads=args.tester.inference.max_thresh_grads
        #self.rid=rid #this is for logging, ignore for now

        #prepare optimization parameters
        self.mu=torch.Tensor([self.args.tester.blind_bwe.optimization.mu[0], self.args.tester.blind_bwe.optimization.mu[1]])
        #self.bt_alpha=self.args.tester.blind_bwe.optimization.alpha
        #self.bt_beta=self.args.tester.blind_bwe.optimization.beta
        #clamping parameters
        self.fcmin=self.args.tester.blind_bwe.fcmin
        if self.args.tester.blind_bwe.fcmax =="nyquist":
                self.fcmax=self.args.exp.sample_rate//2
        else:
                self.fcmax=self.args.tester.blind_bwe.fcmax
        self.Amin=self.args.tester.blind_bwe.Amin
        self.Amax=self.args.tester.blind_bwe.Amax
        #used for congerence checking
        self.tol=self.args.tester.blind_bwe.optimization.tol
        

        self.start_sigma=self.args.tester.posterior_sampling.start_sigma
        if self.start_sigma =="None":
            self.start_sigma=None
        print("start sigma", self.start_sigma)


        



    def update_diff_params(self):
        #the parameters for testing might not be necesarily the same as the ones used for training
        self.diff_params.sigma_min=self.args.tester.diff_params.sigma_min
        self.diff_params.sigma_max =self.args.tester.diff_params.sigma_max
        self.diff_params.ro=self.args.tester.diff_params.ro
        self.diff_params.sigma_data=self.args.tester.diff_params.sigma_data
        #par.diff_params.meters stochastic sampling
        self.diff_params.Schurn=self.args.tester.diff_params.Schurn
        self.diff_params.Stmin=self.args.tester.diff_params.Stmin
        self.diff_params.Stmax=self.args.tester.diff_params.Stmax
        self.diff_params.Snoise=self.args.tester.diff_params.Snoise


    def data_consistency_step_classic(self, x_hat, y, degradation, filter_params=None):
        """
        Simple replacement method, used for inpainting and FIR bwe
        """
        #get reconstruction estimate
        if filter_params is not None:
            den_rec= degradation(x_hat, filter_params)     
        else:
            den_rec= degradation(x_hat)     
        #apply replacment (valid for linear degradations)
        return y+x_hat-den_rec 
    
    def get_rec_grads(self, x_hat, y, x, t_i, degradation, filter_params=None):
        """
        Compute the gradients of the reconstruction error with respect to the input
        """ 

        if self.args.tester.posterior_sampling.SNR_observations !="None":
            snr=10**(self.args.tester.posterior_sampling.SNR_observations/10)
            sigma2_s=torch.var(y, -1)
            sigma=torch.sqrt(sigma2_s/snr).unsqueeze(-1)
            #sigma=torch.tensor([self.args.tester.posterior_sampling.sigma_observations]).unsqueeze(-1).to(y.device)
            #print(y.shape, sigma.shape)
            y+=sigma*torch.randn(y.shape).to(y.device)

        if filter_params is not None:
            den_rec= degradation(x_hat, filter_params) 
        else:
            den_rec= degradation(x_hat) 

        if len(y.shape)==3:
            dim=(1,2)
        elif len(y.shape)==2:
            dim=1


        if self.args.tester.posterior_sampling.norm=="smoothl1":
            norm=torch.nn.functional.smooth_l1_loss(y, den_rec, reduction='sum', beta=self.args.tester.posterior_sampling.smoothl1_beta)
        elif self.args.tester.posterior_sampling.norm=="cosine":
            cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
            norm = (1-cos(den_rec, y)).clamp(min=0)
            print("norm",norm)
        elif self.args.tester.posterior_sampling.stft_distance.use:
            if self.args.tester.posterior_sampling.stft_distance.use_multires:
                print(" applying multires ")
                norm1, norm2=self.norm(y, den_rec)
                norm=norm1+norm2
            elif self.args.tester.posterior_sampling.stft_distance.mag:
                print("logmag", self.args.tester.posterior_sampling.stft_distance.logmag)
                norm=blind_bwe_utils.apply_norm_STFTmag_fweighted(y, den_rec, self.args.tester.posterior_sampling.freq_weighting, self.args.tester.posterior_sampling.stft_distance.nfft, logmag=self.args.tester.posterior_sampling.stft_distance.logmag)
                print("norm", norm)
            else:
                norm=blind_bwe_utils.apply_norm_STFT_fweighted(y, den_rec, self.args.tester.posterior_sampling.freq_weighting, self.args.tester.posterior_sampling.stft_distance.nfft)
        else:
            norm=torch.linalg.norm(y-den_rec,dim=dim, ord=self.args.tester.posterior_sampling.norm)

        
        rec_grads=torch.autograd.grad(outputs=norm.sum(),
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        normguide=torch.linalg.norm(rec_grads)/self.args.exp.audio_len**0.5
        
        #normalize scaling
        s=self.xi/(normguide+1e-6)
        
        #optionally apply a treshold to the gradients
        if False:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        
        return s*rec_grads/t_i

    def get_score_rec_guidance(self, x, y, t_i, degradation, filter_params=None):

        x.requires_grad_()
        x_hat=self.get_denoised_estimate(x, t_i)
        #x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))

        #if self.args.tester.filter_out_cqt_DC_Nyq:
        #    x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)

        #add noise to y

        rec_grads=self.get_rec_grads(x_hat, y, x, t_i, degradation, filter_params)

        #if filter_params is not None:
        #    den_rec= degradation(x_hat, filter_params) 
        #else:
        #    den_rec= degradation(x_hat) 

        #if len(y.shape)==3:
        #    dim=(1,2)
        #elif len(y.shape)==2:
        #    dim=1

        #if self.args.tester.posterior_sampling.norm=="smoothl1":
        #    norm=torch.nn.functional.smooth_l1_loss(y, den_rec, reduction='sum', beta=self.args.tester.posterior_sampling.smoothl1_beta)
        #else:
        #    norm=torch.linalg.norm(y-den_rec,dim=dim, ord=self.args.tester.posterior_sampling.norm)

        
        #rec_grads=torch.autograd.grad(outputs=norm,
        #                              inputs=x)

        #rec_grads=rec_grads[0]
        
        #normguide=torch.linalg.norm(rec_grads)/self.args.exp.audio_len**0.5
        
        #normalize scaling
        #s=self.xi/(normguide*t_i+1e-6)
        
        #optionally apply a treshold to the gradients
        #if False:
        #    #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
        #    rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        
        score=self.denoised2score(x_hat, x, t_i)
        #score=(x_hat.detach()-x)/t_i**2

        #apply scaled guidance to the score
        score=score-rec_grads

        return score
    
    def get_denoised_estimate(self, x, t_i):
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))

        if self.args.tester.filter_out_cqt_DC_Nyq:
            x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
        return x_hat
    

    def get_score(self,x, y, t_i, degradation, filter_params=None):
        if y==None:
            assert degradation==None
            #unconditional sampling
            with torch.no_grad():
                #print("In sampling", x.shape, t_i.shape)
                #print("before denoiser", x.shape)
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                if self.args.tester.filter_out_cqt_DC_Nyq:
                    x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
                score=(x_hat-x)/t_i**2
            return score
        else:
            if self.xi>0:
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i, degradation, filter_params=filter_params)
    
                #optionally apply replacement or consistency step
                if self.data_consistency:
                    #convert score to denoised estimate using Tweedie's formula
                    x_hat=score*t_i**2+x
    
                    try:
                        x_hat=self.data_consistency_step(x_hat)
                    except:
                        x_hat=self.data_consistency_step(x_hat,y, degradation)
    
                    #convert back to score
                    score=(x_hat-x)/t_i**2
    
            else:
                #raise NotImplementedError
                #denoised with replacement method
                with torch.no_grad():
                    x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                        
                    #x_hat=self.data_consistency_step(x_hat,y, degradation)
                    if self.data_consistency:
                        try:
                            x_hat=self.data_consistency_step(x_hat)
                        except:
                            try:
                                x_hat=self.data_consistency_step(x_hat,y, degradation)
                            except:
                                x_hat=self.data_consistency_step(x_hat,y, degradation, filter_params)

        
                    score=(x_hat-x)/t_i**2
    
            return score

    def apply_FIR_filter(self,y):
        y=y.unsqueeze(1)

        #apply the filter with a convolution (it is an FIR)
        y_lpf=torch.nn.functional.conv1d(y,self.filt,padding="same")
        y_lpf=y_lpf.squeeze(1) 

        return y_lpf
    def apply_IIR_filter(self,y):
        y_lpf=torchaudio.functional.lfilter(y, self.a,self.b, clamp=False)
        return y_lpf
    def apply_biquad(self,y):
        y_lpf=torchaudio.functional.biquad(y, self.b0, self.b1, self.b2, self.a0, self.a1, self.a2)
        return y_lpf
    def decimate(self,x):
        return x[...,0:-1:self.factor]

    def resample(self,x):
        N=100
        return torchaudio.functional.resample(x,orig_freq=int(N*self.factor), new_freq=N)

    #def apply_3rdoct_filt(self,x, filt, freq_octs):
    #    filt=f_utils.unnormalize_filter(filt)
    #    y=f_utils.apply_filter(x, filt, self.args.tester.blind_bwe.NFFT,self.args.exp.sample_rate, freq_octs.to(x.device),interpolation="hermite_cubic") 
    #    return y[0]
    def prepare_smooth_mask(self, mask, size=10):
        hann=torch.hann_window(size*2)
        hann_left=hann[0:size]
        hann_right=hann[size::]
        B,N=mask.shape
        mask=mask[0]
        prev=1
        new_mask=mask.clone()
        #print(hann.shape)
        for i in range(len(mask)):
            if mask[i] != prev:
                #print(i, mask.shape, mask[i], prev)
                #transition
                if mask[i]==0:
                   print("apply right")
                   #gap encountered, apply hann right before
                   new_mask[i-size:i]=hann_right
                if mask[i]==1:
                   print("apply left")
                   #gap encountered, apply hann left after
                   new_mask[i:i+size]=hann_left
                #print(mask[i-2*size:i+2*size])
                #print(new_mask[i-2*size:i+2*size])
                
            prev=mask[i]
        return new_mask.unsqueeze(0).expand(B,-1)

    def predict_bwe_AR(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        y_masked,
        filt, #filter Tensor with shape ??
        filt_type,
        rid=False,
        test_filter_fit=False,
        compute_sweep=False,
        mask=None
        ):
        assert mask is not None

        #define the degradation model as a lambda
        if filt_type=="fc_A":
            print("fc_A")
            self.freqs=torch.fft.rfftfreq(self.args.tester.blind_bwe.NFFT, d=1/self.args.exp.sample_rate).to(ylpf.device)
            self.params=filt
            print(self.params)


            y=mask*y_masked+(1-mask)*ylpf

            degradation=lambda x: mask*x +(1-mask)*self.apply_filter_fcA(x, self.params)
        elif filt_type=="firwin":
            self.filt=filt.to(ylpf.device)

            y=mask*y_masked+(1-mask)*ylpf

            degradation=lambda x: mask*x +(1-mask)*self.apply_FIR_filter(x)

            #degradation=lambda x: self.apply_FIR_filter(x)
        else:
           raise NotImplementedError

        if self.args.tester.complete_recording.inpaint_DC:
            smooth_mask=self.prepare_smooth_mask(mask, 50)
            y_smooth_masked=smooth_mask*y_masked

            mask_degradation=lambda x: smooth_mask*x 
            self.data_consistency_step=lambda x_hat: self.data_consistency_step_classic(x_hat,y_smooth_masked, mask_degradation)
            self.data_consistency=True


        return self.predict_conditional(y, degradation, rid, test_filter_fit, compute_sweep)

        
    def predict_bwe(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        filt, #filter Tensor with shape ??
        filt_type,
        rid=False,
        test_filter_fit=False,
        compute_sweep=False
        ):
        print("test_filter_fit", test_filter_fit)
        print("compute_sweep", compute_sweep)

        #define the degradation model as a lambda
        if filt_type=="firwin":
            self.filt=filt.to(ylpf.device)
            degradation=lambda x: self.apply_FIR_filter(x)
        elif filt_type=="firwin_hpf":
            self.filt=filt.to(ylpf.device)
            degradation=lambda x: self.apply_FIR_filter(x)
        elif filt_type=="cheby1":
            b,a=filt
            self.a=torch.Tensor(a).to(ylpf.device)
            self.b=torch.Tensor(b).to(ylpf.device)
            degradation=lambda x: self.apply_IIR_filter(x)
        elif filt_type=="biquad":
            b0, b1, b2, a0, a1, a2=filt
            self.b0=torch.Tensor(b0).to(ylpf.device)
            self.b1=torch.Tensor(b1).to(ylpf.device)
            self.b2=torch.Tensor(b2).to(ylpf.device)
            self.a0=torch.Tensor(a0).to(ylpf.device)
            self.a1=torch.Tensor(a1).to(ylpf.device)
            self.a2=torch.Tensor(a2).to(ylpf.device)
            degradation=lambda x: self.apply_biquad(x)
        elif filt_type=="resample":
            self.factor =filt
            degradation= lambda x: self.resample(x)
            return self.predict_resample(ylpf,(ylpf.shape[0], self.args.exp.audio_len), degradation)
        elif filt_type=="decimate":
            self.factor =filt
            degradation= lambda x: self.decimate(x)
            return self.predict_resample(ylpf,(ylpf.shape[0], self.args.exp.audio_len), degradation)
            #elif filt_type=="3rdoct":
            #    freq_octs=torch.tensor(f_utils.get_third_octave_bands(self.args.exp.sample_rate, fmin=self.args.tester.blind_bwe.range.fmin, fmax=self.args.exp.sample_rate/2))
            #    filt=f_utils.normalize_filter(filt)
            #    degradation= lambda x: self.apply_3rdoct_filt(x, filt, freq_octs)
        elif filt_type=="fc_A":
            print("fc_A")
            self.freqs=torch.fft.rfftfreq(self.args.tester.blind_bwe.NFFT, d=1/self.args.exp.sample_rate).to(ylpf.device)
            self.params=filt
            print(self.params)
            degradation=lambda x:  self.apply_filter_fcA(x, self.params)
        else:
           raise NotImplementedError
        
        if self.data_consistency:
            #normal data consistency
            self.data_consistency_step=lambda x,y,degradation: self.data_consistency_step_classic(x,y, degradation)

        return self.predict_conditional(ylpf, degradation, rid, test_filter_fit, compute_sweep)

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device,
        rid=False
    ):
        self.y=None
        self.degradation=None
        return self.predict(shape, device, rid)

    def predict_resample(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        shape,
        degradation, #lambda function
    ):
        self.degradation=degradation 
        self.y=y
        return self.predict(shape, y.device)


    def predict_conditional(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        degradation, #lambda function
        rid=False,
        test_filter_fit=False,
        compute_sweep=False
    ):
        self.degradation=degradation 

        #if self.args.tester.posterior_sampling.SNR_observations is not None:
        #    SNR=10**(self.args.tester.posterior_sampling.SNR_observations/10)
        #    sigma2_s=torch.var(y, -1)
        #    sigma=torch.sqrt(sigma2_s/SNR)
        #    y+=sigma*torch.randn(y.shape).to(y.device)

        self.y=y
        return self.predict(y.shape, y.device, rid, test_filter_fit, compute_sweep)

    def predict(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device, #lambda function
        rid=False,
        test_filter_fit=False,
        compute_sweep=False
    ):

        if rid:
            data_denoised=torch.zeros((self.nb_steps,shape[0], shape[1]))
            data_score=torch.zeros((self.nb_steps,shape[0], shape[1]))

        if test_filter_fit:
            filter_params=torch.Tensor([self.args.tester.blind_bwe.initial_conditions.fc, self.args.tester.blind_bwe.initial_conditions.A]).to(device)
            if rid:
                data_filters=torch.zeros((self.nb_steps, filter_params.shape[0]))

        if self.start_sigma is None or self.y is None:
            t=self.diff_params.create_schedule(self.nb_steps).to(device)
            x=self.diff_params.sample_prior(shape, t[0]).to(device)
        else:
            #get the noise schedule
            t = self.diff_params.create_schedule_from_initial_t(self.start_sigma,self.nb_steps).to(device)
            #sample from gaussian distribution with sigma_max variance
            x = self.y + self.diff_params.sample_prior(shape,t[0]).to(device)

        #if self.args.tester.bandwidth_extension.sigma_observations>0 and self.y is not None:
        #    self.y=self.y+self.args.tester.bandwidth_extension.sigma_observations*torch.randn_like(self.y)
        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)

        if compute_sweep:
            self.fc_s=torch.logspace(2.5, 4, 15).to(device)
            self.A_s=torch.linspace(-80, -5, 12).to(device)
            if rid:
                data_norms=torch.zeros((self.nb_steps,self.fc_s.shape[0], self.A_s.shape[0]))
                data_grads=torch.zeros((self.nb_steps,self.fc_s.shape[0], self.A_s.shape[0], 2))

        for i in tqdm(range(0, self.nb_steps, 1)):
            #print("sampling step ",i," from ",self.nb_steps)
            x_hat, t_hat=self.move_timestep(x, t[i], gamma[i],self.diff_params.Snoise)

            score=self.get_score(x_hat, self.y, t_hat, self.degradation)    
            if test_filter_fit:
                denoised_estimate=self.score2denoised(score, x_hat, t_hat)
                est_params=self.fit_params(denoised_estimate, self.y,  filter_params)
                ##print("estimated params",est_params.shape)

            if compute_sweep:
                denoised_estimate=self.score2denoised(score, x_hat, t_hat)
                norms, grads=self.compute_sweep(denoised_estimate, self.y)

            d=-t_hat*score

            if rid: 
                data_denoised[i]=self.score2denoised(score, x_hat, t_hat)
                data_score[i]=score
                if test_filter_fit:
                    data_filters[i]=est_params
                if compute_sweep:
                    data_norms[i]=norms
                    data_grads[i]=grads
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, self.y, t_prime, self.degradation)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d

            
        if rid:
            list_out=(x.detach(), data_denoised.detach(), data_score.detach(),t.detach())
            if test_filter_fit:
                list_out=list_out+(data_filters.detach(),)
            if compute_sweep:
                list_out=list_out+(data_norms.detach(), data_grads.detach())
            return list_out
        else:
            return x.detach()



  
    def denoised2score(self,  x_d0, x, t):
        #tweedie's score function
        return (x_d0-x)/t**2
    def score2denoised(self, score, x, t):
        return score*t**2+x

    def move_timestep(self, x, t, gamma, Snoise=1):
        #if gamma_sig[i]==0 this is a deterministic step, make sure it doed not crash
        t_hat=t+gamma*t
        #sample noise, Snoise is 1 by default
        epsilon=torch.randn(x.shape).to(x.device)*Snoise
        #add extra noise
        x_hat=x+((t_hat**2 - t**2)**(1/2))*epsilon
        return x_hat, t_hat

    def apply_filter_fcA(self, x, filter_params):
        H=blind_bwe_utils.design_filter(filter_params[0], filter_params[1], self.freqs)
        return blind_bwe_utils.apply_filter(x, H,self.args.tester.blind_bwe.NFFT)

    def optimizer_func(self, Xden, Y, params):
        """
        Xden: STFT of denoised estimate
        y: observations
        params: parameters of the degradation model (fc, A)
        """

        #print("before design filter", params)
        H=blind_bwe_utils.design_filter(params[0],params[1], self.freqs)
        return blind_bwe_utils.apply_filter_and_norm_STFTmag_fweighted(Xden, Y, H, self.args.tester.posterior_sampling.freq_weighting_filter)

    def fit_params(self, denoised_estimate, y, filter_params):
        #fit the parameters of the degradation model
        #denoised_estimate: denoised estimate of the signal
        #y: observations
        #degradation: degradation function
        #filter_params: initial estimate of parameters of the degradation model

        #return: reestimated parameters of the degradation model

        if self.args.tester.posterior_sampling.SNR_observations !="None":
            snr=10**(self.args.tester.posterior_sampling.SNR_observations/10)
            sigma2_s=torch.var(y, -1)
            sigma=torch.sqrt(sigma2_s/snr).unsqueeze(-1)
            #sigma=torch.tensor([self.args.tester.posterior_sampling.sigma_observations]).unsqueeze(-1).to(y.device)
            #print(y.shape, sigma.shape)
            y+=sigma*torch.randn(y.shape).to(y.device)

        #add noise to the denoised estimate for regularization
        if self.args.tester.blind_bwe.sigma_den_estimate:
            denoised_estimate=denoised_estimate+torch.randn(denoised_estimate.shape).to(denoised_estimate.device)*self.args.tester.blind_bwe.sigma_den_estimate
        


        Xden=blind_bwe_utils.apply_stft(denoised_estimate, self.args.tester.blind_bwe.NFFT)
        Y=blind_bwe_utils.apply_stft(y, self.args.tester.blind_bwe.NFFT)

        func=lambda  params: self.optimizer_func( Xden, Y, params)
        self.mu=self.mu.to(y.device)
        for i in tqdm(range(self.args.tester.blind_bwe.optimization.max_iter)):
            filter_params.requires_grad=True
                #fc.requires_grad=True
            norm=func(filter_params)

            grad=torch.autograd.grad(norm,filter_params,create_graph=True)
            #update params with gradient descent, using backtracking line search
            t=self.mu
            newparams=filter_params-t.unsqueeze(1)*grad[0]
            #newnorm=func(newparams)
            #backtracking line search loop (is this necessary??)
            #while newnorm>norm-t*self.bt_alpha*grad[0].T@grad[0]:
            #    t=t*self.bt_beta
            #    newparams=filter_params-t*grad[0]
            #    newnorm=func(newparams)

            #update with the found step size
            filter_params=newparams

            filter_params.detach_()
            #limit params to help stability
            if self.args.tester.blind_bwe.optimization.clamp_fc:
                    filter_params[0,0]=torch.clamp(filter_params[0,0],min=self.fcmin,max=self.fcmax)
                    for k in range(1,len(filter_params[0])):
                        filter_params[0,k]=torch.clamp(filter_params[0,k],min=filter_params[0,k-1]+1,max=self.fcmax)
            if self.args.tester.blind_bwe.optimization.clamp_A:
                    filter_params[1,0]=torch.clamp(filter_params[1,0],min=self.Amin,max=-1 if self.args.tester.blind_bwe.optimization.only_negative_A else self.Amax)
                    for k in range(1,len(filter_params[0])):
                        filter_params[1,k]=torch.clamp(filter_params[1,k],min=self.Amin,max=filter_params[1,k-1] if self.args.tester.blind_bwe.optimization.only_negative_A else self.Amax)
    

            #checck if params are converging
            #if i>0:
            #   if (torch.abs(filter_params[0]-prev_params[0])<self.tol[0]) and (torch.abs(filter_params[1]-prev_params[1])<self.tol[1]):
            #     break
            if i>0:
                if (torch.abs(filter_params[0]-prev_params[0]).mean()<self.tol[0]) and (torch.abs(filter_params[1]-prev_params[1]).mean()<self.tol[1]):
                     break

            prev_params=filter_params.clone().detach()

        #print("fc: ",filter_params[0].item()," A: ", filter_params[1].item())
        print(filter_params)
        
        return filter_params


    def compute_sweep(self, denoised_estimate, y):

        Xden=blind_bwe_utils.apply_stft(denoised_estimate, self.args.tester.blind_bwe.NFFT)
        Y=blind_bwe_utils.apply_stft(y, self.args.tester.blind_bwe.NFFT)

        func=lambda  params: self.optimizer_func( Xden, Y, params)

        grads=torch.zeros(self.fc_s.shape[0], self.A_s.shape[0], 2)
        norms=torch.zeros(self.fc_s.shape[0], self.A_s.shape[0])
        #iterate over fc and A values

        for fc in range(self.fc_s.shape[0]):
            for A in range(self.A_s.shape[0]):
                #print("fc: ",self.fc_s[fc].item(),"A: ",self.A_s[A].item())
                params=torch.Tensor([self.fc_s[fc], self.A_s[A]]).requires_grad_(True)
                norm=func(params)
                grads[fc,A,:]=torch.autograd.grad(norm,params,create_graph=True)[0]
                norms[fc,A]=norm
        return norms.detach(), grads.detach()


    def predict_blind_bwe(
        self,
        y,  #observations (lowpssed signal) Tensor with shape (L,)
        rid=False,
        compute_sweep=False,
        ):



        
        self.freqs=torch.fft.rfftfreq(self.args.tester.blind_bwe.NFFT, d=1/self.args.exp.sample_rate).to(y.device)
        self.degradation=lambda x, filter_params: self.apply_filter_fcA(x, filter_params)

        if self.data_consistency:
            #normal data consistency
            self.data_consistency_step=lambda x,y,degradation, filter_params: self.data_consistency_step_classic(x,y, degradation, filter_params)

        #get shape and device from the observations tensor
        shape=y.shape
        device=y.device

        #initialise filter parameters
        filter_params=torch.Tensor([self.args.tester.blind_bwe.initial_conditions.fc, self.args.tester.blind_bwe.initial_conditions.A]).to(device)
        if len(filter_params.shape)==1:
            filter_params.unsqueeze_(1)
        print(filter_params.shape)

        shape_filter_params=filter_params.shape #fc and A
        #retrieve the shape from the initial estimate of the parameters
        
        if compute_sweep:
            self.fc_s=torch.logspace(2.5, 4, 15).to(device)
            self.A_s=torch.linspace(-80, -5, 12).to(device)
            if rid:
                data_norms=torch.zeros((self.nb_steps,self.fc_s.shape[0], self.A_s.shape[0]))
                data_grads=torch.zeros((self.nb_steps,self.fc_s.shape[0], self.A_s.shape[0], 2))

        if rid:
            data_denoised=torch.zeros((self.nb_steps,shape[0], shape[1]))
            data_filters=torch.zeros((self.nb_steps,*shape_filter_params))
            print(data_filters.shape)
    


        if self.start_sigma is None:
            t=self.diff_params.create_schedule(self.nb_steps).to(device)
            x=self.diff_params.sample_prior(shape, t[0]).to(device)
        else:
            #get the noise schedule
            t = self.diff_params.create_schedule_from_initial_t(self.start_sigma,self.nb_steps).to(y.device)
            #sample from gaussian distribution with sigma_max variance
            x = y + self.diff_params.sample_prior(shape,t[0]).to(device)

        #if self.args.tester.posterior_sampling.SNR_observations !="none":
        #    snr=10**(self.args.tester.posterior_sampling.SNR_observations/10)
        #    sigma2_s=torch.var(y, -1)
        #    sigma=torch.sqrt(sigma2_s/snr).unsqueeze(-1)
        #    #sigma=torch.tensor([self.args.tester.posterior_sampling.sigma_observations]).unsqueeze(-1).to(y.device)
        #    #print(y.shape, sigma.shape)
        #    y+=sigma*torch.randn(y.shape).to(y.device)

        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)



        for i in tqdm(range(0, self.nb_steps, 1)):
            #print("sampling step ",i," from ",self.nb_steps)
            x_hat, t_hat=self.move_timestep(x, t[i], gamma[i])

            x_hat.requires_grad_(True)

            x_den=self.get_denoised_estimate(x_hat, t_hat)

            x_den_2=x_den.clone().detach()

            filter_params=self.fit_params(x_den_2, y,  filter_params)

            rec_grads=self.get_rec_grads(x_den, y, x_hat, t_hat, self.degradation, filter_params)
            
            x_hat.detach_()

            score=self.denoised2score(x_den_2, x_hat, t_hat)-rec_grads


            if self.args.tester.posterior_sampling.data_consistency:
                #apply data consistency here!
                #it is a bit ugly, but I need to convert the score to denoied estimate again
                x_den_3=self.score2denoised(score, x_hat, t_hat)
                x_den_3=self.data_consistency_step(x_den_3, y, self.degradation, filter_params)
                score=self.denoised2score(x_den_3, x_hat, t_hat)


            if compute_sweep:
                norms, grads=self.compute_sweep(x_den_2, y)

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score

            if rid: 
                data_denoised[i]=x_den_2
                data_filters[i]=filter_params
                if compute_sweep:
                    data_norms[i]=norms
                    data_grads[i]=grads
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                x_prime.requires_grad_(True)

                x_den=self.get_denoised_estimate(x_prime, t_prime)

                x_den_2=x_den.clone().detach()

                filter_params=self.fit_params(x_den_2, y,  filter_params)

                rec_grads=self.get_rec_grads(x_den, y, x_prime, t_prime, self.degradation, filter_params)

                x_prime.detach_()

                score=self.denoised2score(x_den_2, x_prime, t_prime)-rec_grads

                if self.args.tester.posterior_sampling.data_consistency:
                    #apply data consistency here!
                    #it is a bit ugly, but I need to convert the score to denoied estimate again
                    x_den_3=self.score2denoised(score, x_prime, t_prime)
                    x_den_3=self.data_consistency_step(x_den_3, y, self.degradation, filter_params)
                    score=self.denoised2score(x_den_3, x_prime, t_prime)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d

        if rid:
            list_out=(x.detach(), filter_params.detach(), data_denoised.detach(),t.detach(), data_filters.detach())
            if compute_sweep:
                list_out=list_out+(data_norms.detach(), data_grads.detach())
            return list_out
        else:
            return x.detach() , filter_params.detach()


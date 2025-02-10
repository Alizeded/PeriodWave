import math
import torch
from model.base import BaseModule
from model.diffusion_module import SinusoidalPosEmb

import torch.nn as nn
from model.periodwave_utils import MultiPeriodGenerator, MelSpectrogramUpsampler, FinalBlock

LRELU_SLOPE = 0.1
from math import sqrt

class VectorFieldEstimator(BaseModule):
    def __init__(self, n_mel, periods, final_dim=32, hidden_dim=512):
        super(VectorFieldEstimator, self).__init__()     
                                   
        self.len_periods = len(periods)
        self.hidden_dim = hidden_dim
        ### Mel condition
        self.MelCond = MelSpectrogramUpsampler(n_mel, periods, hidden_dim)
        ### Time Condition
        self.time_pos_emb = SinusoidalPosEmb(hidden_dim//2) # We recommend using hidden_dim instead of hidden_dim//2. Please refer the code of encodec version
        self.period_emb = nn.Embedding(self.len_periods, hidden_dim//2) # We recommend using hidden_dim instead of hidden_dim//2. Please refer the code of encodec version
        torch.nn.init.normal_(self.period_emb.weight, 0.0, (hidden_dim//2) ** -0.5) # We recommend using hidden_dim instead of hidden_dim//2. Please refer the code of encodec version

        self.period_token = torch.LongTensor([i for i in range(self.len_periods)]).cuda()
        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim*4),
                                       nn.SiLU(), torch.nn.Linear(hidden_dim*4, hidden_dim))
        
        ### Multi-period Audio U-net 
        self.mpg = MultiPeriodGenerator(periods=periods, final_dim=final_dim, hidden_dim=hidden_dim)

        self.proj_layer = FinalBlock(final_dim)
    def remove_weight_norm(self):
        print('Removing weight norm...')
        self.MelCond.remove_weight_norm()
        self.mpg.remove_weight_norm()
        self.proj_layer.remove_weight_norm()
  
    def forward(self, x, mel, t):

        # Mel condition
        cond = self.MelCond(mel)
        # Time Condition
        t = self.time_pos_emb(t)  

        p = self.period_emb(self.period_token) * math.sqrt(self.hidden_dim) # We accidentally used hidden_dim instead of hidden_dim//2. Although there is no issue in our model, we recommend changing #21, #22, #23 lines following encodec version.
        p = p.unsqueeze(0).expand(t.shape[0], self.len_periods, -1)
        t = torch.concat([t.unsqueeze(1).expand(-1, p.shape[1], -1), p], dim=2)

        t = self.mlp(t)
    
        xs = self.mpg(x, t, cond)

        x = torch.sum(torch.stack(xs), dim=0) / sqrt(self.len_periods)
        x = self.proj_layer(x)

        return x 
    def mel_encoder(self, mel):

        # Mel condition
        mel = self.MelCond(mel)
   
        return mel
     
    def decoder(self, x, cond, t):

        # Time Condition
        t = self.time_pos_emb(t)  

        p = self.period_emb(self.period_token) * math.sqrt(self.hidden_dim)
        p = p.unsqueeze(0).expand(t.shape[0], self.len_periods, -1)
        t = torch.concat([t.unsqueeze(1).expand(-1, p.shape[1], -1), p], dim=2)

        t = self.mlp(t)
    
        xs = self.mpg(x, t, cond)

        x = torch.sum(torch.stack(xs), dim=0) / sqrt(self.len_periods)
        x = self.proj_layer(x)

        return x 
class FlowMatch(BaseModule):
    def __init__(self, n_mel=100, periods=[1,2,3,5,7], noise_scale=0.25, final_dim=32, hidden_dim=512):
        super().__init__() 
        self.sigma_min = 1e-4 
        self.noise_scale = noise_scale 
        self.estimator = VectorFieldEstimator(n_mel, periods, final_dim, hidden_dim)
    
    @torch.no_grad() 
    def forward(self, y, mel, target_std, n_timesteps, temperature=1.0, solver='euler'): 
        y = y.unsqueeze(1)
        z = torch.randn_like(y)*self.noise_scale*target_std *temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mel.device)
        mel = torch.cat([mel, target_std[:, :, ::256]], dim=1) # Concat target std
        mel = self.estimator.mel_encoder(mel)
        if solver=='euler':
            return self.solve_euler(z, t_span=t_span, mel=mel)
        elif solver=='midpoint':
            return self.solve_midpoint(z, t_span=t_span, mel=mel)

    def solve_euler(self, x, t_span, mel): 
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.reshape(1)
        
        sol = []
        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt = self.estimator.decoder(x, mel, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]
    
    def solve_midpoint(self, x, t_span, mel): 
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.reshape(1)
        
        sol = []
        steps = 1
        while steps <= len(t_span) - 1:

            dphi_dt = self.estimator.decoder(x, mel, t)
            half_dt = 0.5 * dt
            x_mid = x + half_dt * dphi_dt

            x = x + dt * self.estimator.decoder(x_mid, mel, t+half_dt)
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]
    
    def compute_loss(self, x1, mel, target_std, length): 
        b, _, t = mel.shape
        mel = torch.cat([mel, target_std[:, :, ::256]], dim=1) # Concat target std

        x_mask = self.sequence_mask(length, x1.size(1)).to(x1.dtype)
    
        t = torch.rand([b, 1, 1], device=mel.device, dtype=mel.dtype)

        x1 = x1.unsqueeze(1)*x_mask
        x0 = torch.randn_like(x1)*self.noise_scale*target_std*x_mask   # B, 1, T
       
        # Flow from x0 to x1
        y = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        # Gradient vector field
        u = x1 - (1 - self.sigma_min) * x0 

        pred = self.estimator(y, mel, t.squeeze())
        loss = self.scaled_mse_loss(pred, u, target_std, x_mask)
        
        return loss
    
    def scaled_mse_loss(self, decoder_output, target, target_std, x_mask):
        # inverse of diagonal matrix is 1/x for each element 
        # from PriorGrad
        sigma_inv = torch.reciprocal(target_std)
        mse_loss = (((decoder_output - target) * sigma_inv) ** 2)
        mse_loss = (mse_loss*x_mask).sum() / x_mask.sum()
        return mse_loss
    
    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)

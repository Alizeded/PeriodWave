import math
import torch
from model.base import BaseModule
from model.diffusion_module import SinusoidalPosEmb
import torch.nn as nn
from model.periodwave_utils_freeu import MultiPeriodGenerator, MelSpectrogramUpsampler, FinalBlock

LRELU_SLOPE = 0.1
from math import sqrt

class GradLogPEstimator(BaseModule):
    def __init__(self, n_mel, periods, final_dim=32, hidden_dim=512):
        super(GradLogPEstimator, self).__init__()     
                                   
        self.len_periods = len(periods)
        self.hidden_dim = hidden_dim
        ### Mel condition
        self.MelCond = MelSpectrogramUpsampler(n_mel, periods, hidden_dim)
        ### Time Condition
        self.time_pos_emb = SinusoidalPosEmb(hidden_dim//2)
        self.period_emb = nn.Embedding(self.len_periods, hidden_dim//2)
        torch.nn.init.normal_(self.period_emb.weight, 0.0, (hidden_dim//2) ** -0.5)

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

        p = self.period_emb(self.period_token) * math.sqrt(self.hidden_dim)
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
    
    def decoder(self, x, cond, t, s_w=1, b_w=1):

        # Time Condition
        t = self.time_pos_emb(t)  

        p = self.period_emb(self.period_token) * math.sqrt(self.hidden_dim)
        p = p.unsqueeze(0).expand(t.shape[0], self.len_periods, -1)
        t = torch.concat([t.unsqueeze(1).expand(-1, p.shape[1], -1), p], dim=2)

        t = self.mlp(t)
    
        xs = self.mpg(x, t, cond, s_w=s_w, b_w=b_w)

        x = torch.sum(torch.stack(xs), dim=0) / sqrt(self.len_periods)
        x = self.proj_layer(x)

        return x  
class FlowMatch(BaseModule):
    def __init__(self, n_mel=100, periods=[2,3,5,7,11], noise_scale=0.25):
        super().__init__() 
        self.sigma_min = 1e-4 
        self.noise_scale = noise_scale 
        self.estimator = GradLogPEstimator(n_mel, periods)
 
    def forward(self, y, mel, target_std, n_timesteps, temperature=1.0, solver='euler', s_w=1, b_w=1): 

        if y.shape[1] != 1:
            y = y.unsqueeze(1)
        z = torch.randn_like(y)*self.noise_scale*target_std *temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mel.device)

        mel = torch.cat([mel, target_std[:, :, ::256]], dim=1) # Concat target std
        mel = self.estimator.mel_encoder(mel)
        if solver=='euler':
            return self.solve_euler(z, t_span=t_span, mel=mel, s_w=s_w, b_w=b_w)
        elif solver=='midpoint':
            return self.solve_midpoint(z, t_span=t_span, mel=mel, s_w=s_w, b_w=b_w)
        
    def solve_euler(self, x, t_span, mel, s_w=1, b_w=1): 
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.reshape(1)
        
        sol = []
        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt = self.estimator.decoder(x, mel, t, s_w=s_w, b_w=b_w)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]
    
    def solve_midpoint(self, x, t_span, mel, s_w=1, b_w=1): 
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.reshape(1)
        
        sol = []
        steps = 1
        while steps <= len(t_span) - 1:

            dphi_dt = self.estimator.decoder(x, mel, t, s_w=s_w, b_w=b_w)
            half_dt = 0.5 * dt
            x_mid = x + half_dt * dphi_dt

            x = x + dt * self.estimator.decoder(x_mid, mel, t+half_dt, s_w=s_w, b_w=b_w)
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    
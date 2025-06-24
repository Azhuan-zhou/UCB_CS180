import torch
import torch.nn as nn
import numpy as np
from .nrf1 import PE
from utils.data import volrend

class NeuralField(nn.Module):
    def __init__(self, input_dim=3, direction_dim=3, hidden_dim=256, L_coord=10, L_dir=4):
        super(NeuralField, self).__init__()
        self.pe_coord = PE(L_coord)
        self.pe_dir = PE(L_dir)

        
        input_coord_dim = input_dim * (2 * L_coord + 1)
        input_dir_dim = direction_dim * (2 * L_dir + 1)

        self.shared_mlp1 = nn.Sequential(
            nn.Linear(input_coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shared_mlp2 = nn.Sequential(nn.Linear(input_coord_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.density_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU() 
        )
        self.color_li = nn.Linear(hidden_dim,hidden_dim)
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim + input_dir_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  
        )

    def render_rays(self,points,rays_d,while_background=False):
        n_samples = points.shape[1]
        rays_d = rays_d.unsqueeze(1).repeat(1,n_samples,1).reshape(-1,3) # (Bxnum_samples, 3)
        points = points.reshape(-1,3) # (Bxnum_samples, 3)
        density, rgb = self.forward(points,rays_d)
        density = density.reshape(-1,n_samples) # (B, num_samples)
        rgb = rgb.reshape(-1,n_samples,3) # (B, num_samples,3)
        output_rgb = volrend(density,rgb,white_background=while_background) # (B, 3)
        return output_rgb
        
        
        
    def forward(self, x, d):
        x_encoded = self.pe_coord(x)  # (N, input_coord_dim)
        d_encoded = self.pe_dir(d)  # (N, input_dir_dim)

        features = self.shared_mlp1(x_encoded)
        features = torch.cat([x_encoded, features], dim=-1)
        features = self.shared_mlp2(features)

        density = self.density_mlp(features).squeeze()  # (N, 1)

        color = self.color_li(features) # (N,256)
        color_input = torch.cat([d_encoded,color], dim=-1)
        rgb = self.color_mlp(color_input)  # (N, 3)

        return density, rgb
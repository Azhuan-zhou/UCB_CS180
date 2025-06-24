import torch
import torch.nn as nn
import numpy as np

class PE(nn.Module):
    def __init__(self, L=10) -> None:
        super(PE,self).__init__()
        self.L = L
    
    def forward(self,coords):
        encoded = [coords]
        for i in range(self.L):
            freq = 2 ** i * np.pi
            encoded.append(torch.sin(freq * coords))
            encoded.append(torch.cos(freq * coords))
        return torch.cat(encoded, dim=-1)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        return self.network(x)
    
class NeuralField(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, L=10):
        super(NeuralField, self).__init__()
        self.pe = PE(L)
        input_dim_mlp = input_dim * L*2 + input_dim
        self.mlp = MLP(input_dim_mlp,hidden_dim=hidden_dim,output_dim=output_dim)

    def forward(self, x):
        x = self.pe(x)
        return self.mlp(x)
    

import os
import numpy as np
import torch
from models.nrf2 import NeuralField
from utils.data import  RaysData_test
from PIL import Image
from torch.utils.data import DataLoader

def save_as_gif(predicted_images, save_path, fps=10):
    duration = int(1000 / fps)
    frames = [Image.fromarray(image) for image in predicted_images] 
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:], 
        duration=duration, 
        loop=0  
    )
    print(f"GIF saved at {save_path}")
    
def sample(white=False):
    image_dir = "./data/lego/lego_200x200.npz"
    dataset = RaysData_test(image_dir)
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=False)
    model = NeuralField(input_dim=3, direction_dim=3, hidden_dim=256, L_coord=10, L_dir=4).cuda()
    checkpoint = torch.load('model_weights_80.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    h, w = dataset.h, dataset.w
    all_outputs = []
    with torch.no_grad():
        for rays_o, rays_d, points in dataloader:
            rays_o, rays_d, points = rays_o.cuda(), rays_d.cuda(), points.cuda()
            outputs = model.render_rays(points=points,rays_d=rays_d,white_background=white)
            all_outputs.append(outputs.detach().cpu())
        all_outputs = torch.cat(all_outputs,dim=0).numpy() # (B, 3)
        predicted_images = all_outputs.reshape(-1,h,w,3)
        predicted_images = (predicted_images * 255).astype(np.uint8)
    save_as_gif(predicted_images,'./results/render_lego_white.gif',30)
    
sample()
sample(white=True)
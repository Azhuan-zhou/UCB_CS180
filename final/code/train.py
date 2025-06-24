import os
import numpy as np
import torch
from torch import nn, optim
from models.nrf1 import NeuralField
from utils.data import create_loader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def psnr_loss(predicted, target):
    mse = torch.mean((predicted - target) ** 2)
    psnr = -10.0 * torch.log10(mse + 1e-8) 
    return -psnr

def infer(model,h,w):
    model.eval()
    with torch.no_grad():
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=-1).float()
        coords_normalized = (coords / torch.tensor([h, w])).cuda()
        predicted_colors = model(coords_normalized).cpu().numpy()
        predicted_image = predicted_colors.reshape(h, w, 3)
        predicted_image = (predicted_image * 255).astype(np.uint8)
    return predicted_image

def train_model(image_dir, num_epochs=3000, learning_rate=1e-2, hidden_dim=256,L=10):
    dataset, dataloader, _ = create_loader(image_dir)
    h,w = dataset.height, dataset.width
    model = NeuralField(input_dim=2, hidden_dim=hidden_dim, output_dim=3,L=L).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    images= []
    checkpoints = num_epochs // 5
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0

        for coords, colors in dataloader:
            coords, colors = coords.cuda(), colors.cuda()

            optimizer.zero_grad()
            outputs = model(coords)
            loss = psnr_loss(outputs, colors)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        if epoch % checkpoints == 0:
            checkpoint_image = infer(model,h,w)
            images.append(checkpoint_image)
            

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        losses.append(-epoch_loss)

    return model, losses, images

def plot_loss_curve(losses, save_path="./results/loss_curve.png"):
    os.makedirs("./results/", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss", color="blue", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Loss curve saved to {save_path}")
    
def save_images(images,save_path='./results/infer.png'):
    image = np.concatenate(images,axis=1)
    save_path = os.path.join(save_path)
    Image.fromarray(image).save(save_path)
    print(f"Intermediate image saved at {save_path}")

        
def main(loss_path,images_path,image_path,num_epochs=3000,learning_rate=1e-2,hidden_dim=256):
    num_epochs=num_epochs
    learning_rate=learning_rate
    hidden_dim=hidden_dim
    model, losses,images = train_model(image_path,num_epochs,learning_rate,hidden_dim)
    plot_loss_curve(losses=losses,save_path=loss_path)
    save_images(images,save_path=images_path)
    
    
if __name__ == "__main__":
    
    main('./results/loss_curve.png','./results/infer.png','./data/test/fox.jpg')
    main('./results/loss_curve_lr(1e-3).png','./results/infer_lr(1e-3).png','./data/test/fox.jpg',learning_rate=1e-3)
    main('./results/loss_curve(128).png','./results/infer_hidden128.png','./data/test/fox.jpg',hidden_dim=128)
    main('./results/loss_curve_sphere.png','./results/infer_sphere.png','./data/test/sphere.png')


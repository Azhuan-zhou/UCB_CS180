import os
import numpy as np
import torch
from torch import nn, optim
from models.nrf2 import NeuralField
from utils.data import create_loader, RaysData_test
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def psnr_loss(predicted, target):
    mse = torch.mean((predicted - target) ** 2)
    psnr = -10.0 * torch.log10(mse + 1e-8) 
    return -psnr

def infer(model,dataloader_val,h,w):
    model.eval()
    all_outputs = []
    epoch_loss = 0.0
    with torch.no_grad():
        for colors, rays_o, rays_d, points in dataloader_val:
            colors, rays_o, rays_d, points = colors.cuda(), rays_o.cuda(), rays_d.cuda(), points.cuda()
            outputs = model.render_rays(points=points,rays_d=rays_d)
            loss = psnr_loss(outputs, colors).item()
            epoch_loss = epoch_loss-loss
            all_outputs.append(outputs.detach().cpu())
        epoch_loss = -epoch_loss / len(dataloader_val)
        all_outputs = torch.cat(all_outputs,dim=0).numpy() # (B, 3)
        num_piexls = w*h
        predicted_image = all_outputs[:num_piexls,:].reshape(h,w,3)
        predicted_image = (predicted_image * 255).astype(np.uint8)
    return loss, predicted_image

def train_model(image_dir, num_epochs=100, learning_rate=1e-2, hidden_dim=256,L=10):
    dataset_val, dataloader_train, dataloader_val = create_loader(image_dir,batch_size=10000,name='part2')
    h,w = dataset_val.h, dataset_val.w
    model = NeuralField(input_dim=3, direction_dim=3, hidden_dim=256, L_coord=10, L_dir=4).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    losses_val = []
    images= []
    checkpoints = num_epochs // 5
    progress = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    for epoch in progress:
        model.train()
        epoch_loss = 0.0
        for colors, rays_o, rays_d, points in dataloader_train:
            colors, rays_o, rays_d, points = colors.cuda(), rays_o.cuda(), rays_d.cuda(), points.cuda()

            optimizer.zero_grad()
            outputs = model.render_rays(points=points, rays_d=rays_d)
            loss = psnr_loss(outputs, colors)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = -epoch_loss / len(dataloader_train)

        loss_val, checkpoint_image = infer(model, dataloader_val, h, w)
        losses_val.append(loss_val)
        losses.append(epoch_loss)

        if epoch % checkpoints == 0:
            images.append(checkpoint_image)
            torch.save(model.state_dict(), f"model_weights_{epoch}.pth")
        if (epoch + 1) == num_epochs:
            images.append(checkpoint_image)
            torch.save(model.state_dict(), "model_weights_final.pth")

        progress.set_description_str(
        f"Epoch {epoch + 1}/{num_epochs}, Loss train: {epoch_loss:.4f}, Loss val: {loss_val:.4f}"
        )

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss train: {epoch_loss:.4f}, Loss val: {loss_val:.4f}")
        

    return model, losses,losses_val, images

def plot_loss_curve(losses, save_path="./results/loss_curve.png"):
    os.makedirs("./results/", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="PSNR", color="blue", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")
    plt.title("PSNR Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"PSNR curve saved to {save_path}")
    
def save_images(images,save_path='./results/infer.png'):
    for i in range(len(images)):
        image = images[i]
        path = save_path.split('.png')[0] + '_{}'.format(i) + '.png'
        Image.fromarray(image).save(path)
        print(f"Intermediate image saved at {path}")

        
def main(loss_path,images_path,num_epochs=100,learning_rate=5e-4,hidden_dim=256):
    image_dir = "/home/shanlins/nrf/tmp/data/lego/lego_200x200.npz" 
    num_epochs=num_epochs
    learning_rate=learning_rate
    hidden_dim=hidden_dim
    model, losses,losses_val,images = train_model(image_dir,num_epochs,learning_rate,hidden_dim)
    plot_loss_curve(losses=losses,save_path=loss_path)
    plot_loss_curve(losses=losses_val,save_path='./results/loss_curve_3d_val.png')
    save_images(images,save_path=images_path)


    
    
    
if __name__ == "__main__":
    main('./results/loss_curve_3d.png','./results/infer_3d.png')


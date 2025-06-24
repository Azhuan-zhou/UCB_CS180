# Import essential modules. Feel free to add whatever you need.
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.optim as optim
import imageio

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upconv(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool =nn.Sequential(nn.AvgPool2d(kernel_size=7),
                                nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.unflatten = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=7, stride=7, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DownConv(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            UpConv(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class TimeConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
    ):
        super().__init__()

        self.initial_conv = ConvBlock(in_channels, num_hiddens)
        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, num_hiddens * 2)

        self.flatten = Flatten()
        self.unflatten = Unflatten(num_hiddens * 2)

        self.fc1_t = FCBlock(1, num_hiddens * 2) 
        self.fc2_t = FCBlock(1, num_hiddens)  

        self.up1 = UpBlock(num_hiddens * 4, num_hiddens)
        self.up2 = UpBlock(num_hiddens * 2, num_hiddens)
        self.final_conv = ConvBlock(2 * num_hiddens, num_hiddens)
        self.final_conv2 = Conv(num_hiddens, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        batch_size = x.shape[0]
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.flatten(x3)
        
        x4 = self.unflatten(x4)
        t1 = self.fc1_t(t.unsqueeze(-1)).reshape(batch_size,-1,1,1)
        x4 = x4 + t1

        x5 = self.up1(torch.cat([x4, x3], dim=1))
        t2 = self.fc2_t(t.unsqueeze(-1)).reshape(batch_size,-1,1,1)
        x5 = x5 + t2
        x6 = self.up2(torch.cat([x5, x2], dim=1))

        out = self.final_conv(torch.cat([x6, x1], dim=1))
        out = self.final_conv2(out)

        return out
    
def ddpm_schedule(beta1: float, beta2: float, num_ts: int) -> dict:
    """Constants for DDPM training and sampling.

    Arguments:
        beta1: float, starting beta value.
        beta2: float, ending beta value.
        num_ts: int, number of timesteps.

    Returns:
        dict with keys:
            betas: linear schedule of betas from beta1 to beta2.
            alphas: 1 - betas.
            alpha_bars: cumulative product of alphas.
    """
    assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."
    
    # Create a linear schedule of betas
    betas = torch.linspace(beta1, beta2, num_ts)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alphas_bars_prev = nn.functional.pad(alpha_bars[:-1], (1, 0), value=1.0)
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_pre": alphas_bars_prev
    }
def ddpm_forward(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    x_0: torch.Tensor,
    num_ts: int,
) -> torch.Tensor:
    """Algorithm 1 of the DDPM paper.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        x_0: (N, C, H, W) input tensor.
        num_ts: int, number of timesteps.

    Returns:
        (,) diffusion loss.
    """
    unet.train()
    device=x_0.device
    batch_size = x_0.shape[0]
    t = torch.randint(0, num_ts, (batch_size,))
    
    alpha_bars_t = ddpm_schedule["alpha_bars"][t].view(-1, 1, 1, 1).to(device)
    
    epsilon = torch.randn_like(x_0)
    
    x_t = torch.sqrt(alpha_bars_t) * x_0 + torch.sqrt(1 - alpha_bars_t) * epsilon
    
    # Pass through UNet
    t_tensor = torch.tensor(t/ (num_ts-1), dtype=torch.float32, device='cuda') 
    epsilon_pred = unet(x_t, t_tensor)
    
    # Compute L2 loss
    loss = nn.functional.mse_loss(epsilon_pred, epsilon)
    
    return loss

def ddpm_sample(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    img_wh: tuple[int, int],
    num_ts: int,
    seed: int = 0,
) -> torch.Tensor:
    """Algorithm 2 of the DDPM paper with classifier-free guidance.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        seed: int, random seed.

    Returns:
        (N, C, H, W) final sample.
    """
    torch.manual_seed(seed)
    x_t = torch.randn((1, 1, *img_wh), device='cuda')
    
    for t in reversed(range(0, num_ts)):
        t_tensor = torch.full((1,), t / (num_ts-1), device='cuda')
        epsilon_pred = unet(x_t, t_tensor)
        
        alpha_bar_t = ddpm_schedule["alpha_bars"][t]
        alpha_bar_prev = ddpm_schedule["alpha_bars_pre"][t]
        beta_t = ddpm_schedule["betas"][t]
        alpha_t = ddpm_schedule["alphas"][t]
        sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sigma_t = torch.sqrt(beta_t)
        
        x_t_minus_1 = (
            sqrt_recip_alpha_t * (x_t - (1 - alpha_t) / sqrt_one_minus_alpha_bar_t * epsilon_pred)
        )
        
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t_minus_1 += sigma_t * noise
        x_t = x_t_minus_1
    
    return x_t

import imageio
def ddpm_sample_with_grid_gif(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    img_wh: tuple[int, int],
    num_ts: int,
    seed: int = 0,
    gif_path: str = 'ddpm_grid_sample.gif',
    grid_size: tuple[int, int] = (4, 10)
) -> torch.Tensor:
    """Modified DDPM sampling function to create a grid GIF.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        seed: int, random seed.
        gif_path: str, path to save the GIF.
        grid_size: (rows, cols) the number of images per row and column.

    Returns:
        (N, C, H, W) final sample.
    """
    torch.manual_seed(seed)
    images = []  # List to hold all images for GIF frames.

    num_images = grid_size[0] * grid_size[1]
    x_t = torch.randn((num_images, 1, *img_wh), device='cuda')
    
    for t in reversed(range(0, num_ts)):
        t_tensor = torch.full((num_images,), t / num_ts, device='cuda')
        epsilon_pred = unet(x_t, t_tensor)
        
        alpha_bar_t = ddpm_schedule["alpha_bars"][t]
        alpha_bar_prev = ddpm_schedule["alpha_bars_pre"][t]
        beta_t = ddpm_schedule["betas"][t]
        alpha_t = ddpm_schedule["alphas"][t]
        sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
        
        x_t_minus_1 = (
            sqrt_recip_alpha_t * (x_t - (1 - alpha_t) / sqrt_one_minus_alpha_bar_t * epsilon_pred)
        )
        
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t_minus_1 += sigma_t * noise
        x_t = x_t_minus_1
        
        # Collect images at specific intervals.
        if t % 10 == 0:
            grid_image = []
            for i in range(grid_size[0]):  # Row-wise collection
                row_images = []
                for j in range(grid_size[1]):  # Col-wise collection
                    idx = i * grid_size[1] + j
                    img = x_t[idx].squeeze().detach().cpu().numpy()
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype('uint8')
                    row_images.append(img)
                grid_image.append(np.hstack(row_images))
            grid_image = np.vstack(grid_image)
            images.append(grid_image)
    
    imageio.mimsave(gif_path, images, fps=30)
    print(f'Grid GIF saved at {gif_path}')
    
class DDPM(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.betas = betas
        self.num_ts = num_ts
        self.p_uncond = p_uncond
        self.ddpm_schedule = ddpm_schedule(betas[0], betas[1], num_ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.

        Returns:
            (,) diffusion loss.
        """
        return ddpm_forward(
            self.unet, self.ddpm_schedule, x, self.num_ts
        )

    @torch.inference_mode()
    def sample(
        self,
        img_wh: tuple[int, int],
        seed: int = 0,
    ):
        return ddpm_sample(
            self.unet, self.ddpm_schedule, img_wh, self.num_ts, seed
        )
    def sample_git(
        self,
        img_wh,
        seed,
        gif_path, 
    ):
        return ddpm_sample_with_grid_gif(
            self.unet, self.ddpm_schedule, img_wh, self.num_ts, seed,gif_path=gif_path
        )
        
model_epoch5 = TimeConditionalUNet(in_channels=1, num_hiddens=64).to('cuda')
DDPM_5 = DDPM(model_epoch5, betas=(1e-4, 0.02), num_ts=300).to('cuda')
DDPM_5.load_state_dict(torch.load('/home/shanlins/diffusion/TimeConditionalUNet_epoch_5.pth'))
DDPM_5.eval()
DDPM_5.sample_git(img_wh=(28, 28), seed=42,gif_path='./image/DDPM_5_sample.gif')
        
model_epoch20 = TimeConditionalUNet(in_channels=1, num_hiddens=64).to('cuda')
DDPM_20 = DDPM(model_epoch20, betas=(1e-4, 0.02), num_ts=300).to('cuda')
msg2 = DDPM_20.load_state_dict(torch.load('/home/shanlins/diffusion/TimeConditionalUNet_epoch_20.pth'))
DDPM_20.eval()
DDPM_20.sample_git(img_wh=(28, 28), seed=42,gif_path='./image/DDPM_20_sample.gif')

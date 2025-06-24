import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pdb
import viser, time

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  
    image = np.array(image) / 255.0 
    return torch.tensor(image, dtype=torch.float32)

class ImageDataset(Dataset):
    def __init__(self, image_path):
        self.images = [load_image(image_path)]
        self.height, self.width,_ = self.images[0].shape

    def __len__(self):
        return len(self.images)

            
    def __getitem__(self, idx):
        return self.images[idx]

def collate_fn1(batch):
    num_samples = 10000 
    coords = []
    colors = []

    for image in batch:
        h, w, _ = image.shape
        x_coords = np.random.randint(0, w, size=num_samples)
        y_coords = np.random.randint(0, h, size=num_samples)

        sampled_coords = np.stack((y_coords, x_coords), axis=-1)
        sampled_colors = image[y_coords, x_coords] 

        coords.append(sampled_coords)
        colors.append(sampled_colors)

    coords = np.concatenate(coords, axis=0)  
    colors = np.concatenate(colors, axis=0) 

    coords = coords / np.array([h, w])

    return torch.tensor(coords, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)

    
def create_loader(image_dir,batch_size=1, name='part1'):
    if name == 'part1':
        dataset_train = ImageDataset(image_dir)
        dataset_val = dataset_train
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1)
        dataloader_val =  DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn1)
        
    elif name == 'part2':
        dataset_train = RaysData(image_dir,'train')
        dataset_val = RaysData(image_dir,'val')
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    return dataset_val, dataloader_train, dataloader_val

def transform(c2w, x_c):
    c2w = c2w.to(x_c.dtype) # Bx4x4
    x_c_homogeneous = torch.cat([x_c, torch.ones(x_c.shape[0], 1, device=x_c.device)], dim=1).unsqueeze(-1)  # Bx4x1
    x_w_homogeneous = (c2w @ x_c_homogeneous).squeeze(-1)
    return x_w_homogeneous[:, :3].to(x_c.dtype)


def pixel_to_camera(K, uv, s):
    bs = uv.shape[0]
    uv_homogeneous = torch.cat([uv, torch.ones(uv.shape[0], 1, device=uv.device)], dim=1).unsqueeze(-1) # Bx3x1
    K_inv = torch.linalg.inv(K).unsqueeze(0).repeat(bs,1,1) # Bx3x3
    x_c_homogeneous = (K_inv @ uv_homogeneous).squeeze(-1) * s.unsqueeze(-1)  # Bx3
    return x_c_homogeneous[:, :3]

def pixel_to_ray(K, c2w, uv):
    ray_o = c2w[:,:3, 3].reshape(-1,3) # Bx3
    
    x_c = pixel_to_camera(K, uv, torch.tensor([1.0], device=uv.device)) # Bx3
    
    x_w = transform(c2w, x_c) # B * 3
    
    ray_d = x_w - ray_o 
    ray_d = ray_d / torch.linalg.norm(ray_d, dim=-1, keepdim=True) # B * 3
    
    return ray_o, ray_d


def sample_rays_from_images(num_images,h,w,camera_intrinsics, camera_extrinsics):
    rays_o_list, rays_d_list, colors_list = [], [], []

    u, v = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="xy",
    )  # (HxW,1)
    uv = torch.stack([u+0.5, v+0.5], dim=-1).reshape(-1, 2) # HxW,2
    K = camera_intrinsics
    for i in range(num_images):
        c2w = camera_extrinsics[i].repeat(h*w,1,1) # HxW,3,3

        ray_o,ray_d = pixel_to_ray(K,c2w,uv) # (HxW,3)  (HxW,3)

        rays_o_list.append(ray_o)
        rays_d_list.append(ray_d)

    rays_o = torch.cat(rays_o_list, dim=0).float() # (HxWxnum_images,3)
    rays_d = torch.cat(rays_d_list, dim=0).float() # (HxWxnum_images,3))

    return rays_o, rays_d

def sample_points_along_rays(ray_o, ray_d, near=2.0, far=6.0, n_samples=64, perturb=True):
    t_vals = torch.linspace(near, far, n_samples).unsqueeze(0).repeat(ray_o.shape[0], 1)  # (N, n_samples)
    if perturb:
        t_width = 0.02
        t_vals = t_vals + (torch.rand_like(t_vals) * t_width)

    points = ray_o.unsqueeze(1) + ray_d.unsqueeze(1) * t_vals.unsqueeze(-1)  # (N, n_samples, 3)
    
    return points, t_vals

def volrend(sigmas, rgbs, step_size=4.0/64, white_background=False):
    if white_background:
        sigmas = torch.cat([sigmas, 9999 * torch.ones_like(sigmas[:, :1])], dim=-1)
        rgbs = torch.cat([rgbs, torch.ones_like(rgbs[:, :1])], dim=-2)
        step_size = torch.cat([step_size, torch.ones_like(step_size[:, :1])], dim=-1)
    alphas = 1.0 - torch.exp(-sigmas * step_size)   #(N, n_sample)

    t = torch.exp(torch.cumsum(-sigmas * step_size, dim=-1)) #(N, n_sample)
    t = torch.cat([torch.ones_like(t[:, :1]), t[:, :-1]], dim=-1)

    weights = (t * alphas).unsqueeze(-1)  #(N, n_sample,1)
    rendered_colors = torch.sum(weights * rgbs, dim=-2) # # (N, 3)
    
    return rendered_colors

class  RaysData(Dataset):
    def __init__(self, image_path,mode):
        data = np.load(image_path)
        if mode == 'train':
            images= torch.tensor(data["images_train"] / 255.0)
            c2ws = torch.tensor(data["c2ws_train"])
            perturb=True
        elif mode == 'val':
            images = torch.tensor(data["images_val"] / 255.0)
            c2ws = torch.tensor(data["c2ws_val"])
            perturb=False
        self.images = images
        self.c2ws = c2ws
        h = images.shape[1]
        w = images.shape[2]
        self.h = h
        self.w = w
        num_images = images.shape[0]
        focal = data["focal"].item()
        self.focal = focal
        K = torch.tensor([[focal,0,w/2],[0,focal,h/2],[0,0,1]])
        self.K = K
        self.colors = images.reshape(-1,3)
        self.rays_o,self.rays_d = sample_rays_from_images(num_images,h,w,K,c2ws) # (B,3) (B,3) (B,3)
        self.points, _ = sample_points_along_rays(self.rays_o,self.rays_d,perturb=perturb) # (B, n_samples, 3)

    def __len__(self):
        return self.colors.shape[0]

    def __getitem__(self, index):
        return (
            self.colors[index],
            self.rays_o[index],
            self.rays_d[index],
            self.points[index]
        )
    def sample_rays(self,N=10000):
        l = self.colors.shape[0]
        index = torch.randint(0, l, (N,))
        return self.rays_o[index], self.rays_d[index], self.colors[index], self.points[index]
  
class  RaysData_test(Dataset):
    def __init__(self, image_path):
        data = np.load(image_path)
        c2ws = torch.tensor(data["c2ws_test"])
        num_images = c2ws.shape[0]
        perturb=False
        self.c2ws = c2ws
        self.h = 200
        self.w = 200
        focal = data["focal"].item()
        self.focal = focal
        K = torch.tensor([[focal,0,self.w/2],[0,focal,self.h/2],[0,0,1]])
        self.K = K
        self.rays_o,self.rays_d = sample_rays_from_images(num_images,self.h,self.w,K,c2ws) # (B,3) (B,3) (B,3)
        self.points, _ = sample_points_along_rays(self.rays_o,self.rays_d,perturb=perturb) # (B, n_samples, 3)

    def __len__(self):
        return self.rays_d.shape[0]

    def __getitem__(self, index):
        return (
            self.rays_o[index],
            self.rays_d[index],
            self.points[index]
        )
    def sample_rays(self,N=10000):
        l = self.rays_d.shape[0]
        index = torch.randint(0, l, (N,))
        return self.rays_o[index], self.rays_d[index], self.points[index]

def test1():
    dataset =  RaysData("/home/shanlins/nrf/tmp/data/lego/lego_200x200.npz",'train')
    rays_o, rays_d, pixels,points = dataset.sample_rays(100)
    points = points.cpu().numpy()
    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    print(rays_d.shape)
    print(rays_d.shape)
    print(points.shape)
    H, W = dataset.images.shape[1:3]
    images = dataset.images.cpu().numpy()
    c2ws=dataset.c2ws.cpu().numpy()

    server = viser.ViserServer(share=True)
    
    for i, (image, c2w) in enumerate(zip(images,c2ws)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, dataset.focal),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
        server.flush()
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )
        server.flush()
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )
    time.sleep(1000)
  
def test2():
    uvs_start = 0
    uvs_end = 40000
    dataset = RaysData("/home/shanlins/nrf/tmp/data/lego/lego_200x200.npz",'train')
    u, v = torch.meshgrid(
        torch.arange(dataset.h, dtype=torch.long),
        torch.arange(dataset.w, dtype=torch.long),
        indexing="xy",
    ) 
    sample_uvs = torch.stack([u, v], dim=-1).reshape(-1, 2)
    # # Uncoment this to display random rays from the first image
    indices = np.random.randint(low=0, high=40_000, size=100)
    #indices_x = np.random.randint(low=100, high=200, size=100)
    #indices_y = np.random.randint(low=0, high=100, size=100)
    #indices = indices_x + (indices_y * 200)
    H, W = dataset.h,dataset.w
    colors, rays_o, rays_d, points = dataset[uvs_start:uvs_end]
    colors = colors.cpu().numpy().astype(np.float32)
    points = points.cpu().numpy()[indices]
    rays_o = rays_o.numpy()[indices]
    rays_d = rays_d.numpy()[indices]
    images = dataset.images.cpu().numpy().astype(np.float32)
    c2ws=dataset.c2ws.cpu().numpy()
    #pdb.set_trace()
    assert np.all(images[0, sample_uvs[:,1], sample_uvs[:,0]] == colors[uvs_start:uvs_end])
    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(images, c2ws)):
      server.add_camera_frustum(
        f"/cameras/{i}",
        fov=2 * np.arctan2(H / 2, dataset.K[0, 0].numpy()),
        aspect=W / H,
        scale=0.15,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
        image=image
      )
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
      positions = np.stack((o, o + d * 6.0))
      server.add_spline_catmull_rom(
          f"/rays/{i}", positions=positions,
      )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.03,
    )
    time.sleep(1000)
    

if __name__ == "__main__":
    test1()
    #test2()
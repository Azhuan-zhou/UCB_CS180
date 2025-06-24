import os 
import numpy as np
import scipy
import json
import skimage.io as io
import skimage as sk
import cv2
from scipy.signal import convolve2d
import math
from scipy.ndimage import distance_transform_edt
images_dir = './images/'

    
def constructA_b(pts1,pts2):
    assert pts1.shape == pts2.shape
    l = pts1.shape[0]
    A = np.zeros((2*l,8))
    b = np.zeros((2*l,1))
    for i in range(l):
        x1,y1 = pts1[i]
        x2,y2 = pts2[i]
        A[2*i] = [x1,y1,1,0,0,0,-x1*x2,-y1*x2]
        A[2*i+1] = [0,0,0,x1,y1,1,-x1*y2,-y1*y2]
        b[2*i,0] = x2   
        b[2*i+1,0] = y2
        
    return A, b

def leastSquares(A:np.ndarray,b:np.ndarray):
    A_T = A.T
    ATA = A_T @ A
    ATA_inv = np.linalg.inv(ATA)
    ATb = A_T @ b
    return ATA_inv @ ATb

def computeH(im1_pts,im2_pts):
    A, b = constructA_b(im1_pts,im2_pts)
    H = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.concatenate([H,np.array([[1.]])],axis=0)
    return H.reshape((3,3))

def computeCanvas(im,H):
    # Get the height and width of the input image
    h, w = im.shape[:2]
    corners = [[0, 0], [h-1, 0], [0, w-1], [h-1, w-1]] # Top-left, top-right, bottom-left, bottom-right
    #Compute the corner coordinates of the image
    max_x = -float('inf')
    max_y = -float('inf')
    min_x = float('inf')
    min_y = float('inf')
    for corner in corners:
        corner.append(1)
        corner = np.array([corner])
    # Warp the corner coordinates using H
        warped_corner = (H @ corner.T)[:,0]
        warped_corner /= warped_corner[2]
        max_x = int(max(max_x, np.ceil(warped_corner[0])))
        max_y = int(max(max_y, np.ceil(warped_corner[1])))
        min_x = int(min(min_x, np.floor(warped_corner[0])))
        min_y = int(min(min_y,np.floor(warped_corner[1])))
    return max_x,min_x,max_y,min_y

def warpImage(im,H):
    max_x, min_x, max_y, min_y = computeCanvas(im,H)
    # Create grid of output image coordinates 
    bound_x = max_x
    bound_y = max_y
    dst_pts = np.array(list(np.ndindex((bound_x,bound_y))))

    # Convert the coordinates to homogeneous coordinates
    output_coords = np.concatenate([dst_pts, np.ones((dst_pts.shape[0],1))], axis=-1).T

    # Apply the inverse homography to get the corresponding points in the original image
    H_inv = np.linalg.inv(H)
    input_coords = H_inv @ output_coords
    input_coords /= input_coords[2]  # Normalize by the third coordinate

    # Extract x and y coordinates for the input image
    input_x = input_coords[0].reshape(-1,1)
    input_y = input_coords[1].reshape(-1,1)
    points = np.array(list(np.ndindex(im.shape[:-1])))
    values = np.array(list(zip(im[:,:,0].flatten(),im[:,:,1].flatten(),im[:,:,2].flatten())))
    # Perform interpolation for all channels at once
    warped_image = scipy.interpolate.griddata(
        points=points,
        values=values,  
        xi=(input_x, input_y),  
        method='linear', 
        fill_value=0 
        )
    warped_image = warped_image.reshape((bound_x,bound_y,3))
    return warped_image

def load_points(path):
    with open(path) as f:
        points =  json.load(f)
        points_1 = np.array(points['im1Points'])[:,[1,0]]
        points_2 = np.array(points['im2Points'])[:,[1,0]]
    return points_1, points_2

def conv(img, filter):
    if len(img.shape) == 3:
        _,_,c = img.shape
        img_list = [img[:,:,i] for i in range(c)]
    elif len(img.shape) == 2:
        img_list = [img]
    results = []
    for img in img_list:
        assert len(img.shape) == len(filter.shape)
        results.append(convolve2d(img,filter,'same'))
    return np.dstack(results)
    
def GaussianStack(image,ksize=40,layers=2):
    stack = [image]
    for i in range(layers):
        sigma = 2 ** i
        D = cv2.getGaussianKernel(ksize,sigma)
        gaussian_filter = D @ D.T
        blur_img = conv(image,gaussian_filter)
        stack.append(blur_img)
    return stack

def LaplacianStack(image,ksize=40,layers=2):
    gaussian_stack = GaussianStack(image,ksize,layers)
    stack = []
    for i in range(len(gaussian_stack)-1):
        pre_blur_img = gaussian_stack[i]
        cur_blur_img = gaussian_stack[i+1]
        stack.append(pre_blur_img-cur_blur_img)
    stack.append(gaussian_stack[-1])
    return stack

def blend(img1,img2,mask1, mask2):
    L_img1_stack = LaplacianStack(img1)
    L_img2_stack = LaplacianStack(img2)
    g_mask_stack1 = GaussianStack(mask1)
    g_mask_stack2 = GaussianStack(mask2)
    
    assert len(L_img1_stack) == len(L_img2_stack) == len(g_mask_stack1) == len(g_mask_stack2)
    collapse = np.zeros_like(L_img1_stack[0]).astype(np.float64)
    for i in range(len(L_img1_stack)):
        collapse = collapse + g_mask_stack1[i] * L_img1_stack[i] + g_mask_stack2[i]*L_img2_stack[i]
    return collapse

def create_mask(im, mask, value, gradient_type='im1'):
    img_mask = np.any(im != 0, axis=2)
    mask_value = np.zeros(im.shape)
    mask_value[img_mask, :] = 1
    
    if gradient_type == 'im1':
        mask_value[mask] = value[mask]
    elif gradient_type == 'im2':
        mask_value[mask] = 1 - value[mask]
    
    return mask_value

def stitch(img1,img2,name=None):
    mask = np.any(img1 != 0, axis=2) & np.any(img2 != 0, axis=2)
    cols = np.where(np.any(mask, axis=0))[0]
    values = np.linspace(1, 0, num=cols.shape[0])
    # Initialize overlap gradient
    overlap_values = np.zeros(img1.shape)
    # Apply gradient to the overlap columns
    for col, val in zip(cols, values):
        overlap_values[mask[:, col], col, :] = val
    # Create masks for both images using a helper function
    im1_mask = create_mask(img1, mask, overlap_values, gradient_type='im1')
    im2_mask = create_mask(img2, mask, overlap_values, gradient_type='im2')
    if name:
        io.imsave(images_dir+'mask_{}_1.png'.format(name),im1_mask)
        io.imsave(images_dir+'mask_{}_2.png'.format(name),im2_mask)
    # Stitch images
    stitched_image = blend(img1,img2,im1_mask,im2_mask)
    return np.clip(stitched_image, 0, 1)

def warp_images_to_ref(images, images_pts):
    center_im = images[0]
    center_im_pts = images_pts[0]
    warped_images = [center_im]
    
    for i in range(1, len(images)):
        img = images[i]
        img_pts = images_pts[i]
        H = computeH(img_pts, center_im_pts)
        warped_im = warpImage(img, H)
        warped_images.append(warped_im)
    
    return warped_images

def project_to_mosaic_plane(warped_images):
    # Determine the size of the output mosaic
    canvas_x = max(img.shape[0] for img in warped_images)
    canvas_y = max(img.shape[1] for img in warped_images)
    
    new_imgs= []
    for img in warped_images:
        new_img = np.zeros((canvas_x, canvas_y, 3))
        new_img[:img.shape[0], :img.shape[1], :] = img
        new_imgs.append(new_img)
    return new_imgs

def create_mosaic(images, images_pts,name):
    # Warp each image to the reference image
    warped_images = warp_images_to_ref(images, images_pts)

    # Project all warped images onto a larger mosaic plane
    new_images = project_to_mosaic_plane(warped_images)

    # Stitch all images into a single final mosaic 
    final_mosaic = new_images[0]
    for i in range(1, len(new_images)):
        final_mosaic = stitch(final_mosaic, new_images[i],name)
    return final_mosaic
    


def rectification_laptop():
    img1 = sk.img_as_float(io.imread(images_dir+"laptop.png"))
    book_down_pts = np.array([[242,243],[140,578],[621,571],[380,969]])
    book_front_pts = np.array([[520,371],[213,371],[520,888],[213,888]])
    H = computeH(book_down_pts,book_front_pts)
    img = warpImage(img1,H)
    img = np.clip(img, 0, 1)  
    img = (img * 255).astype(np.uint8) 
    io.imsave(images_dir+'re1.png',img)
    
def rectification_slide():
    img1 = sk.img_as_float(io.imread(images_dir+"slide.png"))[:,:,:3]
    book_down_pts = np.array([[43,303],[217,809],[641,295],[657,813]])
    book_front_pts = np.array([[89,291],[89,897],[599,291],[599,897]])
    H = computeH(book_down_pts,book_front_pts)
    img = warpImage(img1,H)
    img = np.clip(img, 0, 1)  
    img = (img * 255).astype(np.uint8) 
    io.imsave(images_dir+'re2.png',img)
    
def mosaic_library():
    left = sk.img_as_float(io.imread(images_dir+"library1.png"))[:,:,:3]
    right = sk.img_as_float(io.imread(images_dir+"library2.png"))[:,:,:3]
    points1,points2 = load_points(images_dir+'library1_library2.json')
    images = [left,right]
    pts = [points1,points2]
    img = create_mosaic(images,pts,name='library')
    io.imsave(images_dir+'mosaic1.png', img)
    
def mosaic_lecture():
    left = sk.img_as_float(io.imread(images_dir+"lecture1.png"))[:,:,:3]
    right = sk.img_as_float(io.imread(images_dir+"lecture2.png"))[:,:,:3]
    points1,points2 = load_points(images_dir+'lecture1_lecture2.json')
    images = [left,right]
    pts = [points1,points2]
    img = create_mosaic(images,pts,name='lecture')
    io.imsave(images_dir+'mosaic2.png', img)
    
def mosaic_garden():
    left = sk.img_as_float(io.imread(images_dir+"garden1.png"))[:,:,:3]
    right = sk.img_as_float(io.imread(images_dir+"garden2.png"))[:,:,:3]
    points1,points2 = load_points(images_dir+'garden1_garden2.json')
    images = [left,right]
    pts = [points1,points2]
    img = create_mosaic(images,pts,name='garden')
    io.imsave(images_dir+'mosaic3.png', img)
    
if __name__ == "__main__":
    rectification_laptop()
    rectification_slide()
    mosaic_lecture()
    mosaic_garden()
    mosaic_library()
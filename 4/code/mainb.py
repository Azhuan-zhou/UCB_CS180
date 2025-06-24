import os
import numpy as np
import skimage as sk
from skimage.io import imread, imsave
from harris import get_harris_corners, dist2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy.spatial.distance import cdist
from main import computeH, warpImage, project_to_mosaic_plane,stitch
from tqdm import tqdm
import random

# Adaptive Non-Maximal Suppression
def ANMS(h, coords,c_robust=0.9, n_ip=100):
    num_points = coords.shape[1]
    print(num_points)
    corner_values = h[coords[0], coords[1]] # (num_points,)
    radii = np.full(num_points, np.inf)
    # Calculate the squared distances between all pairs of points
    distances = dist2(coords.T, coords.T)
    for i in range(num_points):
        # Find the distances to all stronger points
        stronger_indices = np.where(corner_values > c_robust * corner_values[i])[0]
        if len(stronger_indices) > 0:
            radii[i] = np.min(distances[i, stronger_indices])
    indexs = np.argsort(radii)[::-1]
    if num_points<n_ip:
        n_ip = num_points
    selected_indices = indexs[:n_ip]
    new_coords = coords[:, selected_indices]
    return new_coords
        
def extract_feature_descriptors(image, keypoints, sigma=1.0):
    descriptors = []
    num_points = keypoints.shape[1]
    for i in range(num_points):
        (y, x) = keypoints[:,i]
        # Check if the window around the keypoint is within image boundaries
        if y - 20 < 0 or y + 20 > image.shape[0] or x - 20 < 0 or x + 20 > image.shape[1]:
            continue  
        # 40x40 window around the keypoint
        patch = image[y - 20:y + 20, x - 20:x + 20]
        smoothed_patch = gaussian_filter(patch, sigma=sigma)
        # Resize the smoothed 40x40 patch to an 8x8 patch
        resized_patch = resize(smoothed_patch, (8, 8), mode='reflect', anti_aliasing=True)

        # Bias/Gain normalization
        mean = np.mean(resized_patch)
        std_dev = np.std(resized_patch)
        if std_dev > 0:
            normalized_patch = (resized_patch - mean) / std_dev
        else:
            normalized_patch = resized_patch - mean
        descriptors.append(normalized_patch.flatten())

    return np.array(descriptors)
     
def feature_matching(des1,des2,threshold=0.3):   
    distances = dist2(des1,des2)
    matches = []
    for i in range(des1.shape[0]):
        distance_i = distances[i]
        sorted_distance_i = np.argsort(distance_i)
        first = distance_i[sorted_distance_i[0]]
        second = distance_i[sorted_distance_i[1]]
        ratio = first / second
        if ratio < threshold:
            best_j = sorted_distance_i[0]
            matches.append((i,best_j))
    return matches

def get_matching_points(coords1,coords2,matches):
    matches1 = []
    matches2 = []
    for i,j in matches:
        matches1.append(coords1[:,i])
        matches2.append(coords2[:,j])
    matches1 = np.array(matches1).T
    matches2 = np.array(matches2).T
    return matches1, matches2

def RANSAC(img1_pts,img2_pts,epochs=1000,epsilon=5):
    # (2,N)
    max_inliers = -1
    max_pts1 = None
    max_pts2 = None
    min_distance = 0
    for i in tqdm(range(epochs),desc='RANSAC'):
        f_pts = random.sample(list(range(img1_pts.shape[1])),4)
        img1_4_pts = img1_pts[:,f_pts]
        img2_4_pts = img2_pts[:,f_pts] # (2,4)
        
        h = computeH(img1_4_pts.T,img2_4_pts.T)
        img1_4_pts_h = np.concatenate([img1_4_pts,np.ones((1,img1_4_pts.shape[1]))],axis=0)
        img1_4_pts_map = h @ img1_4_pts_h
        img1_4_pts_map /= img1_4_pts_map[2]
        img1_4_pts_map =img1_4_pts_map[:2,:]
        num_inliers = 0
        pts1 = []
        pts2 = []
        total_distance = 0
        for j in range(img1_4_pts_map.shape[1]):
            distance = np.sum(np.square(img1_4_pts_map[:,j]-img2_4_pts[:,j]))
            total_distance += distance
            if distance < epsilon:
                num_inliers += 1
                pts1.append(img1_4_pts[:,j])
                pts2.append(img2_4_pts[:,j])
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            min_distance = total_distance
            max_pts1 = pts1
            max_pts2 = pts2
        elif num_inliers == max_inliers:
            if total_distance < min_distance:
                min_distance = total_distance
                max_pts1 = pts1
                max_pts2 = pts2
    max_pts1 = np.array(max_pts1)
    max_pts2 = np.array(max_pts2)
    max_h = computeH(max_pts1,max_pts2)
    return max_h, max_pts1, max_pts2
        
def computeH_auto(img1,img2):
    h1,coords1 = get_harris_corners(img1[:,:,0])
    coords1 = ANMS(h1,coords1)
    des1 = extract_feature_descriptors(img1,coords1)
    
    h2,coords2 = get_harris_corners(img2[:,:,0])
    coords2 = ANMS(h2,coords2)
    des2 = extract_feature_descriptors(img2,coords2)
    
    matches = feature_matching(des1,des2)
    matches1,matches2 = get_matching_points(coords1,coords2,matches)
    max_h,_, _ =RANSAC(matches2,matches1)
    return max_h

def warp_images_to_ref(images):
    center_im = images[0]
    warped_images = [center_im]
    
    for i in range(1, len(images)):
        img = images[i]
        H = computeH_auto(center_im, img)
        print("start warpping image")
        warped_im = warpImage(img, H)
        warped_images.append(warped_im)
    
    return warped_images

def mosaic_auto(images):
    warped_images = warp_images_to_ref(images)

    # Project all warped images onto a larger mosaic plane
    new_images = project_to_mosaic_plane(warped_images)
    # Stitch all images into a single final mosaic 
    final_mosaic = new_images[0]
    print("creating mosaic..........")
    for i in range(1, len(new_images)):
        final_mosaic = stitch(final_mosaic, new_images[i])
    return final_mosaic
    
def step1(path):
    name = path.split('/')[-1].split('.')[0]+'_harris_points'
    img = sk.img_as_float(imread(path))[:,:,:3]
    h, coords = get_harris_corners(img[:,:,0])
    print(coords.shape[1])
    plt.imshow(img)
    plt.plot(coords[1,:], coords[0,:],'ro', markersize=3)
    plt.savefig('./images/{}.png'.format(name), dpi=300, bbox_inches='tight')
    plt.close()
    
def step2(path):
    name = path.split('/')[-1].split('.')[0]+'_harris_points_ANMS'
    img = sk.img_as_float(imread(path))[:,:,:3]
    h, coords = get_harris_corners(img[:,:,0])
    coords_ANMS = ANMS(h,coords)
    plt.imshow(img)
    plt.plot(coords[1,:], coords[0,:],'ro', markersize=3)
    plt.plot(coords_ANMS[1,:], coords_ANMS[0,:],'bo', markersize=3)
    plt.savefig('./images/{}.png'.format(name), dpi=300, bbox_inches='tight')
    plt.close()
    
def step3(path1, path2, name):
    img1 = sk.img_as_float(imread(path1))[:, :, :3]
    h1, coords1 = get_harris_corners(img1[:, :, 0])
    coords1_ANMS = ANMS(h1, coords1, False)
    des1 = extract_feature_descriptors(img1, coords1_ANMS)

    img2 = sk.img_as_float(imread(path2))[:, :, :3]
    h2, coords2 = get_harris_corners(img2[:, :, 0])
    coords2_ANMS = ANMS(h2, coords2, False)
    des2 = extract_feature_descriptors(img2, coords2_ANMS)

    matches = feature_matching(des1, des2)
    matches1, matches2 = get_matching_points(coords1_ANMS, coords2_ANMS, matches)

    # Create a combined image with img1 and img2 side by side
    combined_img = np.concatenate((img1, img2), axis=1)
    
    # Adjust the coordinates of the points in the second image to account for the offset
    matches2_adjusted = matches2.copy()
    matches2_adjusted[1, :] += img1.shape[1]  # Shift the x-coordinates by the width of the first image
    coords2_ANMS_adjusted = coords2_ANMS.copy()
    coords2_ANMS_adjusted[1, :] += img1.shape[1]  # Shift the x-coordinates for ANMS points as well

    # Plot the combined image and draw matching lines
    plt.figure(figsize=(15, 8))
    plt.imshow(combined_img)
    
    # Plot the ANMS points in the first image
    plt.scatter(coords1_ANMS[1, :], coords1_ANMS[0, :], c='r', s=5, marker='o', label='ANMS Points (Image 1)')
    
    # Plot the ANMS points in the second image (adjusted coordinates)
    plt.scatter(coords2_ANMS_adjusted[1, :], coords2_ANMS_adjusted[0, :], c='r', s=5, marker='o', label='ANMS Points (Image 2)')

    # Plot the matching points in the first image
    plt.scatter(matches1[1, :], matches1[0, :], c='b', s=10, marker='o', label='Matches (Image 1)')
    
    # Plot the matching points in the second image (adjusted coordinates)
    plt.scatter(matches2_adjusted[1, :], matches2_adjusted[0, :], c='b', s=10, marker='o', label='Matches (Image 2)')

    # Draw lines connecting the corresponding points
    for i in range(matches1.shape[1]):
        plt.plot([matches1[1, i], matches2_adjusted[1, i]], 
                 [matches1[0, i], matches2_adjusted[0, i]], 'y-', linewidth=.5)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images/{}.png'.format(name), dpi=300, bbox_inches='tight')
    plt.close()
    
def step4_library():
    path1 = './images/library1.png'
    img1 = sk.img_as_float(imread(path1))[:,:,:3]
    path2 = './images/library2.png'
    img2 = sk.img_as_float(imread(path2))[:,:,:3]
    img = mosaic_auto([img1,img2])
    imsave('./images/'+'library_mosaic_auto.png',img)
    
    
def step4_lecture():
    path1 = './images/lecture1.png'
    img1 = sk.img_as_float(imread(path1))[:,:,:3]
    path2 = './images/lecture2.png'
    img2 = sk.img_as_float(imread(path2))[:,:,:3]
    img = mosaic_auto([img1,img2])
    imsave('./images/'+'lecture_mosaic_auto.png',img)
    
def step4_garden():
    path1 = './images/garden1.png'
    img1 = sk.img_as_float(imread(path1))[:,:,:3]
    path2 = './images/garden2.png'
    img2 = sk.img_as_float(imread(path2))[:,:,:3]
    img = mosaic_auto([img1,img2])
    imsave('./images/'+'garden_mosaic_auto.png',img)
    
if __name__ == "__main__":
    step1(path = './images/library1.png')
    step1(path = './images/library2.png')
    step2(path = './images/library1.png')
    step2(path = './images/library2.png')
    step3('./images/library1.png','./images/library2.png',name='library_feature_match')
    step4_library()
    
    step1(path = './images/garden1.png')
    step1(path = './images/garden2.png')
    step2(path = './images/garden1.png')
    step2(path = './images/garden2.png')
    step3('./images/garden1.png','./images/garden2.png',name='garden_feature_match')
    step4_garden()
    step1(path = './images/lecture1.png')
    step1(path = './images/lecture2.png')
    step2(path='./images/lecture1.png')
    step2(path = './images/lecture2.png')
    step3('./images/lecture1.png','./images/lecture2.png',name='lecture_feature_match')
    step4_lecture()
    
    
    
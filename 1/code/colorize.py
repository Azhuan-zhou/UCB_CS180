
# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
import time
from skimage.transform import rescale, resize
import os 
import cv2
from scipy.ndimage import sobel
# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def apply_sobel(channel):
    sobel_x = sobel(channel, axis=0, mode='constant') 
    sobel_y = sobel(channel, axis=1, mode='constant') 
    sobel_edge = np.hypot(sobel_x, sobel_y) 
    return sobel_edge

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,1])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    image = equalized_image.reshape(image.shape)
    return image/255

def apply_histogram_equalization(r,g,b):
    r = histogram_equalization(r)
    g = histogram_equalization(g)
    b = histogram_equalization(b)
    return r,g,b

def check(array,window_size=50):
    results = array.copy()
    l = len(array)
    for i in range(l//2-window_size):
        window = ~array[i+1:i+window_size] # right
        if any(window) and results[i]:
            results[i] = False
    for i in range(l//2, l-window_size): # left
        window = ~array[i:i+window_size-1]
        if any(window):
            results[i+window_size] = False
    return results

def auto_crop(img,threshold1=30,threshold2 = 230):
    image = (img * 255).astype(np.uint8)
    # find non-white pixels
    h,w = image.shape
    border_pixels = np.where((image < threshold1) | (image > threshold2),1,0)
    x = border_pixels.sum(axis=0)
    y = border_pixels.sum(axis=1)
    x = ~(x > 60*h/100)
    x = check(x)
    y = ~(y> 60*w/100)
    y = check(y)
    non_border_x = np.where(x==True)
    non_border_y = np.where(y==True)
    min_x, max_x = np.min(non_border_x), np.max(non_border_x)
    min_y, max_y = np.min(non_border_y), np.max(non_border_y)
    return min_y,max_y,min_x,max_x

def compute_ncc(image1, image2):
    mean1= np.mean(image1)
    mean2= np.mean(image2)
    norm1 = (image1 - mean1) 
    norm2 = (image2 - mean2) 
    numerator = np.sum(norm1 * norm2)
    denominator = np.sqrt(np.sum(norm1**2) * np.sum(norm2**2))
    ncc = numerator / denominator

    return 1-ncc

def comput_sobel(image1,image2):
    image1 = apply_sobel(image1)
    image2 = apply_sobel(image2)
    return compute_ed(image1,image2)

def compute_ed(image1,image2):
    return np.sum((image1 - image2)**2)

def build_pyramid(img, levels):
    pyramid = [img]
    for i in range(levels-1):
        img = rescale(img, 0.5, anti_aliasing=True)
        pyramid.append(img)
    return pyramid

def align_(img1, img2,method,shift=(0,0), max_shift=15):
    best_shift = (0,0)
    best_matching_score = float('inf')
    # probable shifts
    shifts = ((dy, dx) for dy in range(shift[0]-max_shift, shift[0]+max_shift+1) for dx in range(shift[1]-max_shift, shift[1]+max_shift+1))
    for dy,dx in shifts:
        shifted_img = np.roll(img1, shift=(dy, dx), axis=(0, 1))
        if method == 'ed':
            score = compute_ed(shifted_img,img2)
        elif method == 'ncc':
            score = compute_ncc(shifted_img,img2)
        elif method == 'sobel':
            score = comput_sobel(shifted_img,img2)
        else:
            raise ValueError("Unknown type {}".format(method))
        if score < best_matching_score:
            best_matching_score = score
            best_shift = (dy,dx)
    print(best_shift)
    return best_shift

def align(image1,image2,method='sobel',levels=5):
    img_pyramid1 = build_pyramid(image1, levels)
    img_pyramid2 = build_pyramid(image2,levels)

    # Start with the coarsest level
    best_shift = (0, 0) # dy dx
    for level in reversed(range(0,levels)):
        max_shift = 10 - (levels-level-1)*2
        print(max_shift)
        im1 = img_pyramid1[level]
        im2 = img_pyramid2[level]
        best_shift = align_(im1, im2, method=method,max_shift=max_shift,shift=best_shift)
        # Scale up the shift
        best_shift = tuple(s*2 for s in best_shift)
    best_shift = tuple(int(s/2) for s in best_shift)
    print('best shift {}'.format(best_shift))
    return np.roll(image1, shift=best_shift, axis=(0, 1))


def colorize(path):
    
    # name of the input file
    imname = path.split('/')[-1]
    img_name = imname.split('.')[0]
    # read in the image
    im = skio.imread(path)

    # convert to double (might want to do this later on to save     memory)    
    im = sk.img_as_float(im)
    # compute the height of each part (just 1/3 of total)
    height = im.shape[0] // 3

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    stime = time.time()
    r,g,b = apply_histogram_equalization(r,g,b)
    min_y_b,max_y_b,min_x_b,max_x_b = auto_crop(b)
    min_y_g,max_y_g,min_x_g,max_x_g = auto_crop(g)
    min_y_r,max_y_r,min_x_r,max_x_r = auto_crop(r)
    min_x = max(min_x_b,max(min_x_g,min_x_r))
    min_y = max(min_y_b,max(min_y_g,min_y_r))
    max_x = min(max_x_b,min(max_x_g,max_x_r))
    max_y = min(max_y_b,min(max_y_g,max_y_r))
    b = b[min_y:max_y, min_x:max_x]
    g = g[min_y:max_y, min_x:max_x]
    r = r[min_y:max_y, min_x:max_x]
    ag = align(g, b)
    ar = align(r, b)
    # create a color image
    im_out = np.dstack([ar, ag, b])
    im_out = (im_out * 255).astype(np.uint8)
    # save the image
    fname = './test/{}.jpg'.format(img_name)
    skio.imsave(fname, im_out)
    print('color image saved to {}'.format(fname))
    print('Creating a color image consumes {}s'.format(time.time()- stime))
    # display the image
    #skio.imshow(im_out)
    #skio.show()

if __name__ == '__main__':
    path = './data'
    img_paths = [os.path.join(path,i) for i in os.listdir(path)]
    for img_path in img_paths:
        if img_path.endswith('jpg') or img_path.endswith('tif'):
            print('colorize {}'.format(img_path))
            colorize(img_path)
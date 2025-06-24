import numpy as np
from scipy.signal import convolve2d
from skimage import io, color, img_as_ubyte
import os
import cv2
import copy
import scipy as sp
from hybrid_python.align_image_code import align_images, show_img
import matplotlib.pyplot as plt
def save_imgs(imgs:list,names:list):
    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img,name in zip(imgs,names):
        cv2.imwrite(os.path.join(save_dir,name),img)

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
    if len(results) == 1:
        return results[0]
    else:
        return np.dstack(results)

def difference(img:np.ndarray,threshold=20,flag=True):
    D_x = np.array([[1,-1]])
    D_y = D_x.T
    D_x_img = conv(img,D_x)
    D_y_img = conv(img,D_y)
    magnitude = np.sqrt(D_x_img**2 
                        + D_y_img**2)
    binary_mag = (magnitude > threshold)*1.0
    magnitude = rescale(magnitude)
    binary_mag = rescale(binary_mag)
    D_x_img = rescale(D_x_img)
    D_y_img = rescale(D_y_img)
    if flag:
        save_imgs([magnitude,binary_mag,D_x_img,D_y_img],['diff.png','diff_threshold.png','D_x.png','D_y.png'])
    else:
        return magnitude, binary_mag, (D_x_img, D_y_img)

def DOG(img:np.ndarray,threshold:int,ksize=7,sigma=1):
    gaussian_filter_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_filter =  gaussian_filter_1d @ gaussian_filter_1d.T
    # blur image first and then use finite difference operator
    blur_img = conv(img,gaussian_filter)
    DOG_img1,_,(D_x_1,D_y_1) = difference(blur_img,flag=False)
    DOG_img1_bin = (DOG_img1 > threshold)*255
    #  Convolve the gaussian and then convolve with image
    dx = np.array([[1,-1]])
    dy = dx.T
    DOG_x = conv(gaussian_filter,dx)
    DOG_y = conv(gaussian_filter,dy)
    DOG_img_x= conv(img,DOG_x)
    DOG_img_y = conv(img,DOG_y)
    DOG_img2 = np.sqrt(DOG_img_x**2+DOG_img_y**2)
    DOG_img2_bin = (DOG_img2 > threshold)*255
    imgs = [DOG_img1,DOG_img1_bin,D_x_1,D_y_1,DOG_img2,DOG_img2_bin,DOG_img_x,DOG_img_y,cv2.resize(DOG_x,(360,360)),cv2.resize(DOG_y,(360,360))]
    imgs = [rescale(i) for i in imgs]
    save_imgs(imgs,['DOG1.png','DOG1_bin.png','D_x_1.png','D_y_1.png','DOG2.png','DOG2_bin.png','D_x_2.png','D_y_2.png','D_x.png','D_y.png'])

def sharpening(image:np.ndarray,ksize=7,sigma=1):
    alphas = [0,1,2,3,4] # the weight to contorl the percentage of high frequence
    gaussian_kernel_1D = cv2.getGaussianKernel(ksize,sigma)
    gaussian_kernel = gaussian_kernel_1D@gaussian_kernel_1D.T
    original_img_kernel = np.zeros((ksize,ksize))
    for alpha in alphas:
        original_img_kernel[ksize//2][ksize//2] = 1
        unsharp_mask_filter =(alpha+1)*original_img_kernel - alpha*gaussian_kernel
        sharp_img = conv(image,unsharp_mask_filter)
        save_imgs([sharp_img],['sharp_alpha({}).png'.format(alpha)])

def frequency_analysis(image):
    if len(image.shape) == 3:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))))
    plt.show()

def hybrid_imgs(img1,img2,ksize1=20,sigma1=5,ksize2=30,sigma2=20):
    # low frequency
    G1 = cv2.getGaussianKernel(ksize1,sigma1)
    gaussian_kernel_1 = G1@G1.T
    low_frequency_img = conv(img1,gaussian_kernel_1)
    # high frequency
    G2 = cv2.getGaussianKernel(ksize2,sigma2)
    gaussian_kernel_2 = G2@G2.T
    filter1 = np.zeros((ksize2,ksize2))
    filter1[ksize2//2][ksize2//2] = 1
    high_pass_filter = filter1 - gaussian_kernel_2
    high_frequency_img = conv(img2,high_pass_filter)
    # hyrid image: add high frequency and low frequency
    hybrid_img = (low_frequency_img + high_frequency_img) /2 
    # frequency analysis
    imgs = [low_frequency_img,high_frequency_img,hybrid_img]
    imgs = [rescale(i) for i in imgs]
    frequency_analysis(imgs[0])
    frequency_analysis(rescale(img1))
    frequency_analysis(imgs[1])
    frequency_analysis(rescale(img2))
    frequency_analysis(imgs[2])

    save_imgs(imgs,['high_frequency_img.png','low_frequency_img.png','hybrid.png'])

def GaussianStack(image,ksize=20,layers=5):
    stack = [image]
    for i in range(layers):
        sigma = 2 ** i
        D = cv2.getGaussianKernel(ksize,sigma)
        gaussian_filter = D @ D.T
        blur_img = conv(image,gaussian_filter)
        stack.append(blur_img)
    return stack

def LaplacianStack(image,ksize=20,layers=5):
    gaussian_stack = GaussianStack(image,ksize,layers)
    stack = []
    for i in range(len(gaussian_stack)-1):
        pre_blur_img = gaussian_stack[i]
        cur_blur_img = gaussian_stack[i+1]
        stack.append(pre_blur_img-cur_blur_img)
    stack.append(gaussian_stack[-1])
    return stack

def blend(img1,img2,mask):
    L_img1_stack = LaplacianStack(img1)
    L_img2_stack = LaplacianStack(img2)
    g_mask_stack = GaussianStack(mask)
    assert len(L_img1_stack) == len(L_img2_stack) == len(g_mask_stack)
    collapse_imgs = []
    collapse = np.zeros_like(g_mask_stack[0]).astype(np.float64)
    for i in range(len(L_img1_stack)):
        collapse = collapse + g_mask_stack[i] * L_img1_stack[i] + (1-g_mask_stack[i])*L_img2_stack[i]
        collapse_imgs.append(rescale(collapse))
    names = ['collapse{}.png'.format(i) for i in range(1, len(collapse_imgs)+1)]
    save_imgs(collapse_imgs, names)

def rescale(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=-1)
    h,w,c = img.shape
    channels = []
    for i in range(c):
        channel = img[:,:,i]
        min_value = channel.min()
        max_value = channel.max()
        channel = (channel - min_value) / (max_value - min_value)
        channels.append(img_as_ubyte(channel))
    if len(img.shape) == 2:
        return channels[0]
    else:
        return np.dstack(channels)
    
def task_1_1():
    path = './images/cameraman.png'
    img = cv2.imread(path)
    threshold = 80
    difference(img,threshold)

def task_1_2():
    path = './images/cameraman.png'
    img = cv2.imread(path)
    threshold = 30
    DOG(img,threshold)

def task_2_1_1():
    img = cv2.imread('./images/taj.jpg')
    sharpening(img)
def task_2_1_2():
    img = cv2.imread('./images/sunset.jpeg')
    sharpening(img)
def task_2_1_3():
    img = cv2.imread('./images/tower.png')
    G_= cv2.getGaussianKernel(7,1)
    G = G_@G_.T
    img_blur = conv(img,G)
    sharpening(img_blur)
    save_imgs([img_blur],['tower_blur.png']) 

def task_2_2_1():
    # nutmeg and DerekOicture
    img1 = cv2.imread('./images/nutmeg.jpg') 
    img2 = cv2.imread('./images/DerekPicture.jpg')
    img1,img2 = align_images(img1,img2)
    img1 = img_as_ubyte(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) /255
    img2 = img_as_ubyte(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
    hybrid_imgs(img2,img1)
def task_2_2_2():
    # dog and suit
    img1 = cv2.imread('./images/dog.png') 
    img2 = cv2.imread('./images/suit.png')
    img1,img2 = align_images(img1,img2)
    img1 = img_as_ubyte(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) /255
    img2 = img_as_ubyte(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
    hybrid_imgs(img2,img1)
def task_2_2_3():
    # Albert Einstein and Marilyn Monroe
    img1 = cv2.imread('./images/E.png') 
    img2 = cv2.imread('./images/M.png')
    img1,img2 = align_images(img1,img2)
    img1 = img_as_ubyte(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) /255
    img2 = img_as_ubyte(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
    hybrid_imgs(img2,img1)

def task_2_3():
    apple = cv2.imread("./images/apple.jpeg")
    G_s_a = GaussianStack(apple)
    names1 = ['G_a{}.png'.format(i+1) for i in range(6)]
    L_s_a = LaplacianStack(apple)
    names2 = ['L_a{}.png'.format(i+1) for i in range(6)]
    orange = cv2.imread("./images/orange.jpeg")
    G_s_o = GaussianStack(orange)
    names3 = ['G_o{}.png'.format(i+1) for i in range(6)]
    L_s_o = LaplacianStack(orange)
    names4 = ['L_o{}.png'.format(i+1) for i in range(6)]
    images = G_s_a+L_s_a+G_s_o+L_s_o
    images = [rescale(i) for i in images]
    save_imgs(images,names1+names2+names3+names4)

def task_2_4_1():
    # apple and orange
    apple = cv2.imread("./images/apple.jpeg")
    orange = cv2.imread("./images/orange.jpeg")
    mask = np.ones_like(apple)
    mask[:,(apple.shape[1]//2):, :] = 0.0 # vertical
    blend(apple,orange,mask)

def task_2_4_2():
    # suit and dog
    dog = cv2.imread("./images/dog.png")
    suit = cv2.imread("./images/suit.png")
    mask = cv2.imread("./images/mask.png")
    mask = (mask==255)*1.0
    blend(dog,suit,mask)

def task_2_4_3():
    # cat and owl
    cat = cv2.imread("./images/cat.png")
    owl = cv2.imread("./images/owl.jpg")
    mask3 = cv2.imread("./images/mask3.png")
    mask3 = (mask3==255)*1.0
    blend(cat,owl,mask3)

    

if __name__ == '__main__':
    #task_1_1()
    #task_1_2()
    #task_2_1_1()
    #task_2_1_2()
    #task_2_1_3()
    #task_2_2_1()
    #task_2_2_2()
    #task_2_2_3()
    #task_2_3()
    #task_2_4_1()
    #task_2_4_2()
    #task_2_4_3()
    pass

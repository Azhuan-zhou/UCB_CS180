import cv2
import os
import json
import numpy as np
import scipy as sp
import skimage as sk
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
def get_points(path):
    with open(path) as f:
        points =  json.load(f)
        points_1 = np.array(points['im1Points'])[:,[1,0]]
        points_2 = np.array(points['im2Points'])[:,[1,0]]
    return points_1, points_2

def draw_points(img1,img2,points_1,points_2,avg_points,triangulation):
    figure,axes = plt.subplots(1,2,figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].triplot(points_1[:,1],points_1[:,0],triangulation.simplices)
    axes[0].plot(points_1[:,1],points_1[:,0],'o',markersize=1)
    axes[1].imshow(img2)
    axes[1].triplot(points_2[:,1],points_2[:,0],triangulation.simplices)
    axes[1].plot(points_2[:,1],points_2[:,0],'o',markersize=1)
    #plt.show()
    plt.savefig('./images/points.png')

def computeAffine(tri1_pts,tri2_pts):
    # (N*2) 1->origin 2->target
    new_vectors = np.array([[1],[1],[1]])
    tri1_pts = np.concatenate((tri1_pts,new_vectors),axis=-1).T # (3,N)
    tri2_pts = np.concatenate((tri2_pts, new_vectors),axis=-1).T # (3, N)
    tri2_pts_inv = np.linalg.inv(tri2_pts)
    transformation =  tri1_pts @ tri2_pts_inv
    return transformation #(3,3)

def flat_img(img):
    h,w,c = img.shape
    points = list(np.ndindex((h,w)))
    values = zip(img[:,:,0].flatten(),img[:,:,1].flatten(),img[:,:,2].flatten())
    values = list(values)
    return points, values

def mid_way_face(origin_img, origin_tri_pts, target_tri_pts):
    img = np.ones(origin_img.shape) # initialize 
    origin_points, origin_values = flat_img(origin_img)
    color_map = sp.interpolate.NearestNDInterpolator(origin_points,origin_values)
    for origin_tri_pt, target_tri_pt in tqdm(zip(origin_tri_pts, target_tri_pts),total=len(origin_tri_pts), desc="Processing Triangles"):
        transformation = computeAffine(origin_tri_pt, target_tri_pt)
        target_point_1,target_point_2 = sk.draw.polygon(target_tri_pt[:,0],target_tri_pt[:,1]) #(N,2)
        target_points = np.stack((target_point_1,target_point_2,np.ones_like(target_point_1)),axis=0) # 3*N
        target_tranform_points = (transformation @ target_points)
        x, y = target_tranform_points[0], target_tranform_points[1]
        values = color_map((x,y))
        img[target_point_1,target_point_2] = values
    return img

def save_imgs(imgs,names):
    save_dir = './images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    imgs = [sk.img_as_ubyte(img) for img in imgs]
    for i, img in enumerate(imgs):
        path = os.path.join(save_dir,names[i])
    
        sk.io.imsave(path, img)

def get_triangle_points(points,triangles):
    triangle_points = []
    for simplice in triangles.simplices:
        tmp = np.stack(points[simplice],axis=0)
        triangle_points.append(tmp)
    return triangle_points

def morph(im1, im2, im1_pts, im2_pts,triangulation, warp_frac, dissolve_frac):
    mid_pts =warp_frac*im1_pts + (1- warp_frac) * im2_pts
    triangles1 = get_triangle_points(im1_pts,triangulation)
    triangles2 = get_triangle_points(im2_pts,triangulation)
    triangles_mid = get_triangle_points(mid_pts,triangulation)
    midWayFace1 = mid_way_face(im1, triangles1,triangles_mid)
    midWayFace2 = mid_way_face(im2, triangles2, triangles_mid)
    midWayFace = dissolve_frac * midWayFace1 + (1-dissolve_frac) * midWayFace2
    return midWayFace

def morphSequences(im1, im2, im1_pts, im2_pts, num_frames):
    avg_points = 0.5 * im1_pts + 0.5 * im2_pts
    triangulation = sp.spatial.Delaunay(avg_points)
    sequences = []
    for i in range(1,num_frames+1):
        warp_frac = dissolve_frac = (1- i / num_frames)
        sequences.append(morph(im1, im2, im1_pts, im2_pts,triangulation,warp_frac,dissolve_frac))
    return sequences

def correspondences(dir):
    height = 480
    width = 640
    corr_pts = {}
    for filename in os.listdir(dir):
        if filename.endswith("1m.asf") or filename.endswith("1f.asf"):
            with open(os.path.join(dir,filename),'r') as f:
                txt = f.read()
                data = np.array([s.split('\t') for s in txt.split('\n')[16:-6]])[:,2:4].astype(float)
                data[:,0], data[:,1] = data[:,0]*width, data[:,1]*height
                data = np.round(data).astype(int)
                data = np.vstack([data, [0,0], [0,height-1],[width-1,0], [width-1,height-1]])
                data[:,[0,1]] = data[:,[1,0]]
                corr_pts[filename[:-4]] = data
    return corr_pts
def task1():
    img1 = sk.io.imread('./images/resize_me.jpg') /255
    img2 = sk.io.imread('./images/george.jpg') /255
    points1,points2 = get_points('images/resize_me_george.json')
    avg_points = (points1+points2) /2 
    triangulation = sp.spatial.Delaunay(avg_points) 
    draw_points(img1,img2,points1,points2,avg_points,triangulation)

def task2():
    img1 = sk.io.imread('./images/resize_me.jpg') /255
    img2 = sk.io.imread('./images/george.jpg') /255
    points1,points2 = get_points('images/resize_me_george.json')
    avg_points = (points1+points2) /2 
    triangulation = sp.spatial.Delaunay(avg_points) 
    tri1 = get_triangle_points(points1,triangulation)
    tri2 = get_triangle_points(points2,triangulation)
    tri_avg = get_triangle_points(avg_points,triangulation)
    mid_img1 = mid_way_face(img1,tri1,tri_avg)
    mid_img2 = mid_way_face(img2,tri2,tri_avg)
    img = (mid_img1+mid_img2) / 2
    save_imgs([img],['mid_way_me_george.png'])


def task3():
    img1 = sk.io.imread('./images/resize_me.jpg') /255
    img2 = sk.io.imread('./images/george.jpg') /255
    points1,points2 = get_points('images/resize_me_george.json')
    resutls = morphSequences(img1,img2,points1,points2,80)
    resutls = [sk.img_as_ubyte(img) for img in resutls]
    #names = ['{}.png'.format(i) for i in range(len(resutls))]
    #save_imgs(resutls,names)

    imageio.mimsave('./images/sequences.gif',resutls, duration=0.05)
    #imageio.mimsave('./images/sequences.mp4', resutls, fps=20)

def task4_1():
    dirName = './imm_face_db'    
    corr = correspondences(dirName)
    corr_mean = np.round(np.mean([corr[name] for name in corr.keys()],axis=0)).astype(int)
    triangulation = sp.spatial.Delaunay(corr_mean) 
    corr_mean_pts = get_triangle_points(corr_mean,triangulation)
    imgs = []
    imgs_f = []
    imgs_m = []
    example = []
    origin = []
    counts = 0
    for file in tqdm(os.listdir(dirName),total=len(os.listdir(dirName)),desc='Processing:'):
        if file.endswith("1m.jpg") or file.endswith("1f.jpg"):
            name = file.split('.')[0]
            pts = corr[name]
            img = sk.io.imread(os.path.join(dirName,name+'.jpg')) / 255
            tri_pts = get_triangle_points(pts,triangulation)
            img_morph = mid_way_face(img,tri_pts,corr_mean_pts)
            imgs.append(img_morph)
            if counts < 4:
                example.append(img_morph)
                origin.append(img)
                counts += 1
            if file.endswith("1m.jpg"):
                imgs_m.append(img_morph)
            if file.endswith("1f.jpg"):
                imgs_f.append(img_morph)
    mean_faces = np.mean(imgs,axis=0)
    mean_faces_f = np.mean(imgs_f,axis=0)
    mean_faces_m = np.mean(imgs_m,axis=0)
    save_imgs(example,['1.png','2.png','3.png','4.png'])
    save_imgs(origin,['1_o.png','2_o.png','3_o.png','4_o.png'])
    save_imgs([mean_faces,mean_faces_f,mean_faces_m],['mean.png','mean_f.png','mean_m.png'])

    
def task4_2():
    img1 = sk.io.imread('./images/resize_me_2.png') /255
    img2 = sk.io.imread('./images/mean.png') / 255
    points1,points2 = get_points('images/resize_me_2_mean.json')
    im1_tri =  sp.spatial.Delaunay(points1)
    im2_tri = sp.spatial.Delaunay(points2)
    # 1 
    im1_tri_ptrs = get_triangle_points(points1,im2_tri)
    im2_tri_ptrs = get_triangle_points(points2,im2_tri)
    im1_morph = mid_way_face(img1,im1_tri_ptrs,im2_tri_ptrs)
    # 2 
    im1_tri_ptrs = get_triangle_points(points1,im1_tri)
    im2_tri_ptrs = get_triangle_points(points2,im1_tri)
    im2_morph = mid_way_face(img2,im2_tri_ptrs,im1_tri_ptrs)
    save_imgs([im1_morph,im2_morph],['me2mean.png','mean2me.png'])
    
def task5(alpha=-0.5):
    img1 = sk.io.imread('./images/resize_me_2.png') /255
    img2 = sk.io.imread('./images/mean_m.png') / 255
    points1,points2 = get_points('images/resize_me_2_mean.json')
    diff_points = points1 - points2
    new_points = points1 + alpha*diff_points
    new_tri = sp.spatial.Delaunay(new_points)
    im1_tri_ptrs = get_triangle_points(points1,new_tri)
    new_tri_ptrs = get_triangle_points(new_points,new_tri)
    img = mid_way_face(img1,im1_tri_ptrs,new_tri_ptrs)
    save_imgs([img],['Caricature_{}.png'.format(alpha)])

def BellsWhistles():
    img1 = sk.io.imread('./images/resize_me_3.png') /255
    img2 = sk.io.imread('./images/female.jpg') / 255
    points1,points2 = get_points('images/resize_me_3_female.json')
    points_avg = (points1+points2) / 2
    tri_avg = sp.spatial.Delaunay(points_avg)
    im1_tri =  sp.spatial.Delaunay(points1)
    im2_tri = sp.spatial.Delaunay(points2)
    # change shape
    im1_tri_ptrs = get_triangle_points(points1,im2_tri)
    im2_tri_ptrs = get_triangle_points(points2,im2_tri)
    im1_new_shape = mid_way_face(img1,im1_tri_ptrs,im2_tri_ptrs)
    # change appearance
    im1_tri_ptrs = get_triangle_points(points1,im1_tri)
    im2_tri_ptrs = get_triangle_points(points2,im1_tri)
    im2_morph = mid_way_face(img2,im2_tri_ptrs,im1_tri_ptrs)
    im1_new_appearance = 0.4 * img1 + 0.6 * im2_morph
    # change both
    triangulation = sp.spatial.Delaunay(0.5*points1+0.5*points2)
    im1_new_both = morph(img1,img2,points1,points2,triangulation,0.5,0.4)
    save_imgs([im1_new_shape,im1_new_appearance,im1_new_both],['change_shape.png','change_appearance.png','change_both.png'])


if __name__ == '__main__':
    #task1()
    #task2()
    #task3()
    #task4_1()
    #task4_2()
    #task5()
    #BellsWhistles()
    pass
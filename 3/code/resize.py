import skimage as sk

a = sk.io.imread('/Users/azhuan/Documents/code/Python/UCB/CS180/Azhuan-zhou.github.io/3/code/images/resize_me_3.png')
b = sk.io.imread('/Users/azhuan/Documents/code/Python/UCB/CS180/Azhuan-zhou.github.io/3/code/images/female.jpg')
shape = b.shape[:2]
a = sk.transform.resize(a,shape,anti_aliasing=True)
resized_image = (a * 255).astype('uint8')[:,:,:3]

# 保存或显示调整后的图像
sk.io.imsave('./images/resize_me_3.png', resized_image)
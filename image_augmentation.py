import cv2
import numpy as np
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
import os

def anticlockwise_rotation(img):
    angle = 45
    return rotate(img,angle=angle)

def clockwise_rotation(img):
    angle = 45
    return rotate(img,-angle)

def horizontal_flip(img):
    return  np.fliplr(img)

def vertical_flip(img):
    return np.flipud(img)

def random_noise(img):
    return random_noise(img)

def blur_image(img):
    return cv2.GaussianBlur(img, (6,6),0)

def contrast_brightness_control(img):
    alpha = 1   #from 1.0-3.0
    beta = 1.5  #from 0-100
    return cv2.convertScaleAbs(image,alpha=alpha,beta=beta)

img = cv2.imread("/home/bang/Desktop/Image_Processing/image/input_augmentation/cat.jpg",cv2.IMREAD_UNCHANGED)
print(np.shape(img))

output_dir = "/home/bang/Desktop/Image_Processing/image/output_augmentation/"

img_anticlockwise_rotation = anticlockwise_rotation(img)
cv2.imwrite(os.path.join(output_dir,"output_anticlockwise_rotation.jpg"),img_anticlockwise_rotation)

img_clockwise_rotation = clockwise_rotation(img)
cv2.imwrite(os.path.join(output_dir,"output_clockwise_rotation.jpg"),img_clockwise_rotation)

img_horizontal_flip = horizontal_flip(img)
cv2.imwrite(os.path.join(output_dir,"output_horizontal_flip.jpg"),img_horizontal_flip)

img_vertical_flip = vertical_flip(img)
cv2.imwrite(os.path.join(output_dir,"output_vertical_flip.jpg"),img_vertical_flip)

img_add_noise = random_noise(img)
cv2.imwrite(os.path.join(output_dir,"output_add_noise.jpg"),img_add_noise)

img_blur_image = blur_image(img)
cv2.imwrite(os.path.join(output_dir,"output_blur_image.jpg"),img_blur_image)

img_contrast_brightness_control = contrast_brightness_control(img)
cv2.imwrite(os.path.join(output_dir,"output_contrast_brightness_control.jpg"),img_contrast_brightness_control)

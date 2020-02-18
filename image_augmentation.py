import cv2
import numpy as np
from skimage import rotate, AffineTransform, warp
from skimage.util import random_noise
import os

def anticlockwise_rotation(img):
    angle = 45
    return rotate(img,angle)

def clockwise_rotation(img):
    angle = 45
    return rotate(img,-angle)

def horizontal_flip(img):
    return  np.fliplr(img)

def vertical_flip(img):
    return np.flipud(img)

def add_noise(img):
    return random_noise(img)

def blur_image(img):
    return cv2.GaussianBlur(img, (6,6),0)

img = cv2.imread("/home/bang/Desktop/Image_Processing/image/input_augmentation/cat.jpg",cv2.IMREAD_UNCHANGED)

output_dir = "/home/bang/Desktop/Image_Processing/image/output_augmentation/"

img_anticlockwise_rotation = anticlockwise_rotation(img)
cv2.imwrite(os.path.join(output_dir,"output_anticlockwise_rotation.jpg"),img_anticlockwise_rotation)

img_clockwise_rotation = clockwise_rotation(img)


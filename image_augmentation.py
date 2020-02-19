import cv2
import numpy as np
import os
import math
import glob

'''
    Image rotation
    -> rotate_bound() function return a rotated image with some blank gap
    -> rotate_max_area() function return a croped rotated image with no blank gap
'''
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def rotate_max_area(image, angle):
    """
        image: cv2 image matrix object
        angle: in degree
    """
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                    math.radians(angle))
    rotated = rotate_bound(image, angle)
    h, w, _ = rotated.shape
    y1 = h//2 - int(hr/2)
    y2 = y1 + int(hr)
    x1 = w//2 - int(wr/2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]

def rotatedRectWithMaxArea(w, h, angle): 
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long:
    # half constrained case: two crop corners touch the longer side,
    # the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

def horizontal_flip(img):
    return np.fliplr(img)

def vertical_flip(img):
    return np.flipud(img)

def noise_generator (noise_type,image):
    '''
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    '''
    row,col,ch= image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = 0.0001
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss * 255
        gauss = gauss.reshape(row,col,ch)
        #print(gauss)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        noisy = np.random.poisson(image)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss * 0.5
        return noisy
    else:
        return image

def blur_image(img):
    return cv2.blur(img, (10,10))


def contrast_brightness_control(img):
    alpha = 1.5  #from 1.0-3.0
    beta = 1.5  #from 0-100
    return cv2.convertScaleAbs(img,alpha=alpha,beta=beta)

#You can replace path with your own
img_path = (glob.glob("/home/bang/Desktop/Image_Processing/image/input_augmentation/*"))

#Output directory
output_dir = "/home/bang/Desktop/Image_Processing/image/output_augmentation/"

for path in img_path:
    name = path.split("/")[-1].split(".")[0]
    img = cv2 .imread(path,cv2.IMREAD_UNCHANGED)
    rotated = rotate_max_area(img,-45)
    cv2.imwrite(os.path.join(output_dir,"output_rotated.png"),rotated)
    img_horizontal_flip = horizontal_flip(img)
    cv2.imwrite(os.path.join(output_dir,"output_horizontal_flip.png"),img_horizontal_flip)
    img_vertical_flip = vertical_flip(img)
    cv2.imwrite(os.path.join(output_dir,"output_vertical_flip.png"),img_vertical_flip)
    img_add_noise = noise_generator("gauss",img)
    cv2.imwrite(os.path.join(output_dir,"output_add_noise_gauss.png"),img_add_noise)
    img_blur_image = blur_image(img)
    cv2.imwrite(os.path.join(output_dir,"output_blur_image.png"),img_blur_image)
    img_contrast_brightness_control = contrast_brightness_control(img)
    cv2.imwrite(os.path.join(output_dir,"output_contrast_brightness_control.png"),img_contrast_brightness_control)

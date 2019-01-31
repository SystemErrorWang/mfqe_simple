import numpy as np
import cv2

def bgr2yuv(image):
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.147*r - 0.289*g + 0.436*b
    v = 0.615*r - 0.515*g - 0.1*b
    #return np.stack((y, u, v), 2)
    return y, u, v

def yuv2bgr(image):
    y, u, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    b = y + 2.03*u
    g = y - 0.39*u - 0.58*v
    r = y + 1.14*v
    #return np.stack((b, g, r), 2)
    return b, g, r
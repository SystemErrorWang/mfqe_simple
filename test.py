import torch
import cv2
import os
import skvideo
skvideo.setFFmpegPath("C:/Program Files/ffmpeg/bin")
import skvideo.io
import numpy as np

from tqdm import tqdm
from network.MC_subnet import MotionCompensateSubnet
from network.QE_subnet import QualityEnhanceSubnet


def process_image(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    h, w = np.shape(yuv)[:2]
    y_channel = yuv[:, :, 0].reshape(1, 1, h, w)/255.0
    y_channel = torch.tensor(y_channel).float().cuda()
    color = yuv[:, :, 1:]
    return y_channel, color


def train_wrapper(image):
    h, w = np.shape(image)
    image = image.reshape(1, 1, h, w)/255.0
    image = torch.tensor(image).float().cuda()
    return image
    

def read_video(video_path):
    video = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while (cap.isOpened()):
        res, frame = cap.read()
        if type(frame) != type(None):
            video.append(frame)
        else:
            break
    return video, fps


def bgr2yuv(image):
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.169*r - 0.331*g +0.5*b + 128
    v = 0.5*r - 0.419*g -0.081*b + 128
    #return np.stack((y, u, v), 2)
    return y, u, v


def rgb2yuv(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.169*r - 0.331*g +0.5*b + 128
    v = 0.5*r - 0.419*g -0.081*b + 128
    #return np.stack((y, u, v), 2)
    return y, u, v


def yuv2bgr(image):
    y, u, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    b = y + 2.03211*(u - 128)
    g = y - 0.39465*(u - 128) - 0.58060*(v - 128)
    r = y + 1.13983*(v - 128)
    #return np.stack((b, g, r), 2)
    return b, g, r


def rescale(image0, image1):
    '''
    image0: image as the target
    image1: image to be rescaled
    '''
    minb0, maxb0 = np.min(image0[:, :, 0]), np.max(image0[:, :, 0])
    ming0, maxg0 = np.min(image0[:, :, 1]), np.max(image0[:, :, 1])
    minr0, maxr0 = np.min(image0[:, :, 2]), np.max(image0[:, :, 2])
    
    minb1, maxb1 = np.min(image1[:, :, 0]), np.max(image1[:, :, 0])
    ming1, maxg1 = np.min(image1[:, :, 1]), np.max(image1[:, :, 1])
    minr1, maxr1 = np.min(image1[:, :, 2]), np.max(image1[:, :, 2])
    
    image1[:, :, 0] = (image1[:, :, 0]-minb1)/(maxb1-minb1)
    image1[:, :, 0] = image1[:, :, 0]*(maxb0-minb0)+minb0
    
    image1[:, :, 1] = (image1[:, :, 1]-ming1)/(maxg1-ming1)
    image1[:, :, 1] = image1[:, :, 1]*(maxg0-ming0)+ming0
    
    image1[:, :, 2] = (image1[:, :, 2]-minr1)/(maxr1-minr1)
    image1[:, :, 2] = image1[:, :, 2]*(maxr0-minr0)+minr0
    
    return image1.astype(np.uint8)


mc_weight = 'weight\\mcnet_joint_8th_epoch.pth'
qe_weight = 'weight\\qenet_joint_8th_epoch.pth'
video_folder = 'C:\\Users\\Administrator\\Downloads\\h264_2'
name_list = os.listdir(video_folder)
video_path = os.path.join(video_folder, name_list[0])
video, fps = read_video(video_path)


h, w = np.shape(video[0])[:2]
length = len(video)
mcnet = MotionCompensateSubnet().cuda()
qenet = QualityEnhanceSubnet().cuda()
mcnet.load_state_dict(torch.load(mc_weight))
qenet.load_state_dict(torch.load(qe_weight))
codec = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('output.avi', codec, fps, (w, h))

for idx in tqdm(range(length)):
    before = video[(max(0, idx-3))]
    now = video[idx]
    after = video[min(length-1, idx+3)]
    
    before_y, _, _ = bgr2yuv(before)
    now_y, u, v = bgr2yuv(now)
    after_y, _, _ = bgr2yuv(after)
    
    before_y = train_wrapper(before_y)
    now_y = train_wrapper(now_y)
    after_y = train_wrapper(after_y)
    
    
    mc1 = mcnet(now_y, before_y)
    mc2 = mcnet(now_y, after_y)
    qe = qenet(mc1, now_y, mc2)
    qe = qe.detach().cpu().numpy().squeeze()*255.0
    
    out = np.stack((qe, u, v), 2)
    out = np.stack((yuv2bgr(out)), 2)
    out = rescale(now, out)
    out_video.write(out)
    
out_video.release()
cv2.destroyAllWindows()



    
    
        
    
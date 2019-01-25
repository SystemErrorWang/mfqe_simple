import cv2
import os
from tqdm import tqdm
from multiprocessing import Process, Pool

video_folder = '../dataset/original_videos'
split_folder = '../dataset/split'

if not os.path.exists(split_folder):
    os.mkdir(split_folder)

#for i in range(25):
def convert(i):
    video_path = os.path.join(video_folder, 'original_{}.mov'.format(i))
    sub_folder = os.path.join(split_folder, 'split{}'.format(i))
    os.mkdir(sub_folder)
    cap = cv2.VideoCapture(video_path)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(video_len)):
        res, frame = cap.read()
        if type(frame) == type(None):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
        image_path = os.path.join(sub_folder, 
                    'frame{}.bmp'.format("%04d" %frame_idx))
        cv2.imwrite(image_path, frame)
        #frame_idx += 1
        
    
if __name__=='__main__':
    '''
    pool = Pool(processes=2)
    pool.map(convert, range(2))
    '''
    for i in range(2):
        convert(i)
    
    
    
    
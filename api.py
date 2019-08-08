import io
import os 
import cv2
import sys
import time 
import math
import numpy as np 
from PIL import Image
import time
from skimage.measure import compare_ssim



def get_model_api():
    # initialization 
    def SSIM(im, im2):
        return compare_ssim(im, im2)

    def PSNR(im, im2):
        h, w, _ = im.shape 
        #height = im.shape[0]
        #width = im.shape[1]

        R = im[:,:,0]-im2[:,:,0]
        G = im[:,:,1]-im2[:,:,1]
        B = im[:,:,2]-im2[:,:,2]
    
        mser = R*R
        mseg = G*G
        mseb = B*B
        SUM = mser.sum() + mseg.sum() + mseb.sum()
        MSE = SUM / (h * w * 3)
        p = 10*math.log ( (255.0*255.0/(MSE+1e-8)) ,10)
        return p

    def model_api(submissionPath):
        TS = submissionPath
        GT = 'images'

        ssim, psnr = [], []
        for i in range(1,11):
            a = os.path.join(TS, '{}.jpg'.format(i))
            b = os.path.join(GT, '{}.jpg'.format(i))
            imga = cv2.imread(a,0)
            imgb = cv2.imread(b,0)

            """
            try:
                print(imga.shape)
            except AttributeError:
                print('Image {} does not exist.'.format(a))
            print(imgb.shape)

            assert(imga.shape==imgb.shape)
            """
            s = SSIM(imga, imgb)
            ssim.append(s)

            imga = np.array(Image.open(a),'f')
            imgb = np.array(Image.open(b),'f')

            p = PSNR(imga, imgb)
            psnr.append(p)
    
              
        S = sum(ssim)/len(ssim)
        P = sum(psnr)/len(psnr)
        return (S*P)
    
    return model_api

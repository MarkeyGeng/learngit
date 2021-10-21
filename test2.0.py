import cv2 as cv
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader as Dataloader
import os
from PIL import Image
import torch
import random
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


N = 10
testDir = './UTKFace/part3/'

dataTransform = transforms.Compose([
    transforms.ToTensor()
])

def test():
    model = torch.load('./age_gender_model.pt')
    # print(model)
    files = random.sample(os.listdir(testDir), N)
    imgs = []
    imgsData = []
    for file in files:
        img = cv.imread(testDir + file)
        imgs.append(img)
        img_data = cv.resize(img, (224, 224))
        img_data = dataTransform(img_data)
        
        # img_data = (np.float32(img)/255.0 - 0.5) / 0.5
        res = np.zeros(img_data.shape, dtype=np.float32)
        
        cv.normalize(img, res, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        img_data = res
        
        # img_data = img_data.transpose((2, 0, 1))
        
        img_data = torch.from_numpy(img_data)
        imgsData.append(img_data)
        
        

    imgsData = torch.stack(imgsData)

    with torch.no_grad():
        out = model(imgsData.cuda())
        out = out.cpu().detach().numpy() * 110

    for inx in range(N):
        plt.figure()
        plt.suptitle('age:{}'.format(out[inx]))
        plt.imshow(imgs[inx])
        plt.savefig('/home/gjt/img/{}.png'.format(inx))
    plt.show()
    

if __name__ == '__main__':
    test()


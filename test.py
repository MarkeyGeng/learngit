import torch
import os
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import loadData
import matplotlib.pyplot as plt
from PIL import Image

np.set_printoptions(suppress=True)

testDataDir = './data/testset1/'
modelFile = './model/model.pth'

N = 10

def test():
    model = models.resnet18(num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(modelFile), strict=False)
    model.eval()

    files = random.sample(os.listdir(testDataDir), N)
    imgs = []
    imgsData = []
    for file in files:
        img = Image.open(testDataDir + file)
        img_data = loadData.dataTransform(img)
        imgs.append(img)
        imgsData.append(img_data)
    imgsData = torch.stack(imgsData)

    with torch.no_grad():
        out = model(imgsData)
    out = F.softmax(out, dim=1)
    out = out.data.cpu().numpy()

    for inx in range(N):
        plt.figure()
        if out[inx, 0] > out[inx, 1]:
            plt.suptitle('female:{:.1%}, male:{:.1%}'.format(out[inx, 0], out[inx, 1]))
        else:
            plt.suptitle('male:{:.1%}, female:{:.1%}'.format(out[inx, 1], out[inx, 0]))
        plt.imshow(imgs[inx], cmap="gray")
    plt.show()

if __name__ == '__main__':
    test()
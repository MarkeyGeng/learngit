import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import cv2 as cv

def dataSet(dir):
    listImg = []
    ageLabel = []
    for file in os.listdir(dir):
        listImg.append(dir + file)
        ageLabel.append(int(file.split('.')[0].split('-')[1]))
    return listImg, ageLabel

vgg16_model = models.vgg16(pretrained=True)
structure = torch.nn.Sequential(*list(vgg16_model.children())[:])
print(structure)
print(vgg16_model._modules.keys())

new_classifier = torch.nn.Sequential(*list(vgg16_model.children())[-1][:4])
#print('new classifier:', new_classifier)

vgg_model_4096 = models.vgg16(pretrained=True)

vgg_model_4096.classifier = new_classifier

#print(torch.nn.Sequential(*list(vgg_model_4096.children())[:]))

vgg16_model.eval()
vgg_model_4096.eval()

file_dir = './data/testset/7-37-1.png'
# file_dir = './data/testset/2-34-1.png'
IMG_SIZE = 224
im = Image.open(file_dir)

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transRgb = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transHui = transforms.Compose([
    transforms.Normalize(mean=[0.485,], std=[0.229,])
])

im = trans(im)

if im.shape[0] == 1:
    im = im.repeat(3, 1, 1)
    im = transHui(im)
else:
    im = transRgb(im)

im.unsqueeze_(dim=0)

im_feature_1000 = vgg16_model(im).data[0]
im_feature_4096 = vgg_model_4096(im).data[0]
print(im_feature_1000.shape)
print(im_feature_4096.shape)

img = cv.imread('./data/testset/7-37-1.png')
print(img.shape)
(h, w, c) = img.shape
print('{},{},{}'.format(h,w,c))

import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

IMAGE_SIZE = 200

dataTransform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize([0.485, ], [0.229, ])
])

class AgeAndGender(data.Dataset):
    def __init__(self, mode, dir):
        self.mode = mode
        self.listImage = []
        self.ageLabel = []
        self.genderLabel = []
        self.dataSize = 0
        self.transform = dataTransform
        dir = dir + 'trainset1/'

        for file in os.listdir(dir):
            self.listImage.append(dir + file)
            self.dataSize += 1
            self.ageLabel.append(int(file.split('.')[0].split('-')[1]))
            self.genderLabel.append(int(file.split('.')[0].split('-')[2]))
                

    def __getitem__(self, index):
        if self.mode == 'age':
            img = Image.open(self.listImage[index])
            age = self.ageLabel[index]
            return self.transform(img), torch.LongTensor([age])
        elif self.mode == 'gender':
            img = Image.open(self.listImage[index])
            gender = self.genderLabel[index]
            return self.transform(img), torch.LongTensor([gender])
    
    def __len__(self):
        return self.dataSize
                

# def loadFun():
#     trainFileList = os.listdir('data/trainset')
#     n = len(trainFileList)
#     genders = []
#     ages = []
#     for i in range(n):
#         fileName = trainFileList[i]
#         fileName = os.path.splitext(fileName)[0]
#         gender = int(fileName[-1])
#         genders.append(gender)
#         fileName = fileName[:-2]
#         age = int(fileName.split('-')[-1])
#         ages.append(age)  


# if __name__ == '__main__':
#     loadFun()
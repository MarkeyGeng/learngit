import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import torch.nn as nn
from torch.utils.data import DataLoader as DataLoader
import numpy as np
import cv2 as cv
import torch.utils.data as data

MAX_AGE = 110
num_epochs = 1

class AgeGenderDataset(data.Dataset):
    def __init__(self, dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        img_file = os.listdir(dir)
        imgNum = len(img_file)
        self.ages = []
        self.imgs = []
        index = 0
        for file in img_file:
            age = file.split('_')[0]
            self.ages.append(np.float(age)/MAX_AGE)
            self.imgs.append(dir+file)
            index += 1
            if len(self.imgs) >= 1000:
                break;
    
    def __len__(self):
        return len(self.imgs)

    def getNum(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            img_path = self.imgs[idx]
        else:
            img_path = self.imgs[idx]
        
        img = cv.imread(img_path)   # BGR顺序
        h, w, c = img.shape
        # rescale
        img = cv.resize(img, (224, 224))
        img = (np.float32(img)/255.0 - 0.5) / 0.5
        # h, w, c to c, h, w
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img), 'age': self.ages[idx]}
        return sample

def train():
    ds = AgeGenderDataset('./data/UTKFace/part1/')
    num_train_sample = ds.getNum()
    batch_size = 16
    dataloader = DataLoader(ds, batch_size=16, shuffle=True)

    # 用vgg16模型
    model = models.vgg16()
    model.classifier = torch.nn.Sequential(*list(model.children())[-1][:], torch.nn.Sigmoid(), torch.nn.Dropout(p=0.5), torch.nn.Linear(1000, 1))
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    model.train()

    loss_fun = torch.nn.MSELoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batch in enumerate(dataloader):
            imgs_batch, age_batch = sample_batch['image'], sample_batch['age']
            imgs_batch, age_batch = imgs_batch.cuda(), age_batch.cuda()
            optimizer.zero_grad()
            age_batch = age_batch.float()

            # forward
            age_out = model(imgs_batch)
            age_batch = age_batch.view(-1, 1)
    
            # calculate loss
            loss = loss_fun(age_out, age_batch)
            loss.backward()
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 100 == 0:
                print('step: {} \tTraining Loss: {:.6f}'.format(index, loss.item()))
            index += 1

        # 计算平均损失
        train_loss = train_loss / num_train_sample

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining loss: {:.6f}'.format(epoch, train_loss))

    # save model
    model.eval()
    torch.save(model, 'age_gender_model.pt')

if __name__ == '__main__':
    train()
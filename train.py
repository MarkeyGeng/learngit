from loadData import AgeAndGender as AAG
import torch
from torch.utils.data import DataLoader as DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models

dataSetDir = './data/'
model_cp = './model/'

workers = 16        #同时运行的线程
batchSize = 16
lr = 0.0001
nepoch = 1

def train():
    dataFile = AAG('gender', dataSetDir)     #加载数据集
    dataLoader = DataLoader(dataFile, batch_size=16, shuffle=True, num_workers=workers)
    print('Dataset loaded! length of train set is {0}'.format(len(dataFile)))
    
    model = models.resnet18(num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for epoch in range(nepoch):
        for img, gender in dataLoader:
            img, gender = Variable(img), Variable(gender)
            out = model(img)
            loss = criterion(out, gender.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batchSize, loss / batchSize))
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))

if __name__ == '__main__':
    train()
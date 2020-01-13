import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torchvision
import os
import numpy as np
import time
import utils

# import cleverhans.attacks as attack
DOWNLOAD_MNIST = True
test_dataset = datasets.MNIST(
    root='./mnist',
    train=False,  # download test data
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
img = test_dataset.data
label = test_dataset.targets
img = torch.unsqueeze(img, 1).to(torch.float)

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, kernel_size=3, stride=1, padding=1),
            # input shape(1*28*28),(28+1*2-3)/1+1=28 卷积后输出（6*28*28）
            # 输出图像大小计算公式:(n*n像素的图）(n+2p-k)/s+1
            nn.ReLU(True),  # 激活函数
            nn.MaxPool2d(2, 2),  # 28/2=14 池化后（6*14*14）
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # (14-5)/1+1=10 卷积后（16*10*10）
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 池化后（16*5*5）=400，the input of full connection
        )
        self.fc = nn.Sequential(  # full connection layers.
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out


vanillaCNN = torch.load(utils.vanishPATH)
vanilla = CNN(1, 10)
if not torch.cuda.is_available():       #是否可用GPU计算
     vanilla = vanilla.cuda()
vanilla.load_state_dict(vanillaCNN)
batchSize=50
correctImg=[]
correctLabel=[]
correctGrad=[]
criterion = nn.CrossEntropyLoss()
for iter in range(len(img)//batchSize):
    tempImg=Variable(img[iter*batchSize:(iter+1)*batchSize],requires_grad=True)
    tempLabel=Variable(label[iter*batchSize:(iter+1)*batchSize])
    predictA=vanilla(tempImg)
    predict=predictA.argmax(1)
    tempLoss=criterion(predictA,tempLabel)
    #print(tempLoss)
    tempGrad=autograd.grad(tempLoss,tempImg)[0]
    #print(tempGrad[0].shape)
    tempCorrect=np.where(predict==tempLabel)[0]
    #print(predict)
    #time.sleep(1000)
    cI=tempImg[tempCorrect].detach().numpy()
    cL=tempLabel[tempCorrect].detach().numpy()
    cG=tempGrad[tempCorrect].detach().numpy()
    correctImg.extend(cI)
    correctLabel.extend(cL)
    correctGrad.extend(cG)
    #print(len(cI))
print("vanilla correct Img num:{} acc:{}".format(len(correctLabel),len(correctLabel)/len(label)))
correctImg=torch.Tensor(correctImg)
correctLabel=torch.Tensor(correctLabel)
correctGrad=torch.Tensor(correctGrad)


correctAttackImg=[]
correctAttackLabel=[]
sigma=0.01
for iter in range(len(correctImg)//batchSize):
    tempImg = Variable(correctImg[iter * batchSize:(iter + 1) * batchSize])
    tempGrad=Variable(correctGrad[iter * batchSize:(iter + 1) * batchSize])
    tempImg+=sigma*torch.sign(tempGrad)
    tempLabel = Variable(correctLabel[iter * batchSize:(iter + 1) * batchSize])
    predictA = vanilla(tempImg)
    predict = predictA.argmax(1)
    #tempLoss = criterion(predictA, tempLabel)
    # print(tempLoss)
    #tempGrad = autograd.grad(tempLoss, tempImg)[0]
    # print(tempGrad[0].shape)
    tempCorrect = np.where(predict == tempLabel)[0]
    # print(predict)
    # time.sleep(1000)
    cI = tempImg[tempCorrect]
    cL = tempLabel[tempCorrect]
    #cG = tempGrad[tempCorrect]
    correctAttackImg.extend(cI)
    correctAttackLabel.extend(cL)
    #correctGrad.extend(cG)
print("after sigma :{} FGSM correct:{}".format(sigma,len(correctAttackLabel)/len(correctLabel)))

punishCNN = torch.load(utils.punishPATH)
punish = CNN(1, 10)
if not torch.cuda.is_available():       #是否可用GPU计算
     punish = punish.cuda()
punish.load_state_dict(vanillaCNN)
batchSize=50
correctImgv2=[]
correctLabelv2=[]
correctGradv2=[]
for iter in range(len(img)//batchSize):
    tempImg = Variable(img[iter * batchSize:(iter + 1) * batchSize], requires_grad=True)
    tempLabel = Variable(label[iter * batchSize:(iter + 1) * batchSize])
    predictA = punish(tempImg)
    predict = predictA.argmax(1)
    tempLoss = criterion(predictA, tempLabel)
    # print(tempLoss)
    tempGrad = autograd.grad(tempLoss, tempImg)[0]
    # print(tempGrad[0].shape)
    tempCorrect = np.where(predict == tempLabel)[0]
    # print(predict)
    # time.sleep(1000)
    cI = tempImg[tempCorrect].detach().numpy()
    cL = tempLabel[tempCorrect].detach().numpy()
    cG = tempGrad[tempCorrect].detach().numpy()
    correctImgv2.extend(cI)
    correctLabelv2.extend(cL)
    correctGradv2.extend(cG)
print("punish:correct Img num:{} acc:{}".format(len(correctLabel),len(correctLabel)/len(label)))
correctImgv2=torch.Tensor(correctImgv2)
correctLabelv2=torch.Tensor(correctLabelv2)
correctGradv2=torch.Tensor(correctGradv2)
correctAttackImgv2=[]
correctAttackLabelv2=[]
for iter in range(len(correctImgv2)//batchSize):
    tempImg = Variable(correctImgv2[iter * batchSize:(iter + 1) * batchSize])
    tempGrad=Variable(correctGradv2[iter * batchSize:(iter + 1) * batchSize])
    tempImg+=sigma*torch.sign(tempGrad)
    tempLabel = Variable(correctLabelv2[iter * batchSize:(iter + 1) * batchSize])
    predictA = punish(tempImg)
    predict = predictA.argmax(1)
    #tempLoss = criterion(predictA, tempLabel)
    # print(tempLoss)
    #tempGrad = autograd.grad(tempLoss, tempImg)[0]
    # print(tempGrad[0].shape)
    tempCorrect = np.where(predict == tempLabel)[0]
    # print(predict)
    # time.sleep(1000)
    cI = tempImg[tempCorrect]
    cL = tempLabel[tempCorrect]
    #cG = tempGrad[tempCorrect]
    correctAttackImgv2.extend(cI)
    correctAttackLabelv2.extend(cL)
    #correctGrad.extend(cG)
print("after sigma :{} FGSM correct:{}".format(sigma,len(correctAttackLabel)/len(correctLabel)))
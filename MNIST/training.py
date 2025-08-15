## WGAN-GP for MNIST dataset. Models
## are initialized from the standard
## normal distribution having 0 mean
## 0.02 standard deviation Keep Adam
## optimizer to standard betas.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

from tqdm import tqdm

## Hyper-parameters
NOISE = 256
LEARNING_RATE = 0.0002
BATCH_SIZE = 64
EPOCH = 300
CRITIC_ITER = 5
LAMBDA = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(10,256)
        chn = [128,64,32,16]

        self.fc1_1 = self.linear(512,chn[0]*2*2)
        self.conv1 = self.t_conv(chn[0],chn[1])
        self.conv2 = self.t_conv(chn[1],chn[2])
        self.conv3 = self.t_conv(chn[2],chn[3])
        self.conv4 = self.t_conv(chn[3],1,last=True)

    def t_conv(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.Tanh())
        else:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.BatchNorm2d(o), nn.ReLU())
    def linear(self,i,o):
        return nn.Sequential(nn.Linear(i,o),
               nn.BatchNorm1d(o),nn.ReLU())
    
    def forward(self,z,y):
        y = self.embed(y)

        input = torch.cat([z,y],dim=1)
        input = self.fc1_1(input)
        
        input = input.view(-1,128,2,2)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        return input

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        inp = 1
        chn = [16,32,64,128]
        self.embed = nn.Embedding(10,1024) ## 32x32

        self.conv1 = self.conv(inp+1 ,chn[0])
        self.conv2 = self.conv(chn[0],chn[1])
        self.conv3 = self.conv(chn[1],chn[2])
        self.conv4 = self.conv(chn[2],chn[3])

        self.layer = nn.Sequential(nn.Flatten(),
                     nn.Linear(chn[3]*2*2,1))
    
    def conv(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1), nn.LeakyReLU(0.2))
    
    def forward(self,x,y):
        y = self.embed(y).view(-1,1,32,32)
        input = torch.cat([x,y],dim=1)

        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)

        input = self.layer(input)
        return input

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight.data,0.0,0.02)
            nn.init.constant_(m.bias.data,0.0)

def penalty(Cnet,image,fakes,label):
    B,C,H,W = image.shape

    alpha = torch.rand(B,1,1,1).repeat(1,C,H,W)
    alpha = alpha.to(DEVICE)

    interpolated_image = (alpha*image)+((1-alpha)*fakes) ## real+fake
    interpolated_score = Cnet(interpolated_image, label)

    gradient = torch.autograd.grad(
        outputs=interpolated_score,
        inputs =interpolated_image,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm-1)**2)

tranform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])])

datasets = MNIST('MNIST/dataset',True,transform=tranform,download=True)
dataload = DataLoader(datasets,BATCH_SIZE,shuffle=True)

Gnet = Generator().to(DEVICE)
Cnet = Critic().to(DEVICE)

weight_initialization(Gnet)
weight_initialization(Cnet)

Goptim = Adam(Gnet.parameters(), lr=LEARNING_RATE)
Coptim = Adam(Cnet.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    Cnet.train()
    Gnet.train()

    for image,label in tqdm(dataload):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        for xxx in range(CRITIC_ITER):
            Coptim.zero_grad()

            noise = torch.randn(image.shape[0],NOISE)
            noise = noise.to(DEVICE)
            
            fakes = Gnet(noise,label)

            real_logit = Cnet(image,label)
            fake_logit = Cnet(fakes,label)
            gp = penalty(Cnet,image,fakes,label)

            Closs = -(real_logit.mean()-fake_logit.mean()) + LAMBDA*gp
            
            Closs.backward(retain_graph=True)
            Coptim.step()

        Goptim.zero_grad()
        Gloss = -Cnet(fakes,label).mean()
        Gloss.backward()
        Goptim.step()
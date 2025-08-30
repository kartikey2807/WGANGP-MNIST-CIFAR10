## WGAN-GP implementation for Toy Dataset

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torchsummary import summary
from torch.utils.data import DataLoader,Dataset
from sklearn.datasets import make_circles

from tqdm import tqdm

## Hyper-parameters
NOISE = 4
IMP = 0.6
LR = 0.00001
BATCH_SIZE = 256
EPOCHS = 35000
CRITIC_ITERS = 5
LAMBDA_GP = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_1 = self.linear(6,8)
        self.fc2_1 = self.linear(8,4)
        self.fc3_1 = self.linear(4,4)
        self.fc4_1 = self.linear(4,2,last=True)

        self.embed = nn.Embedding(2,2)
    
    def linear(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.Linear(i,o),nn.Tanh())
        else:
            return nn.Sequential(nn.Linear(i,o),
                   nn.BatchNorm1d(o),nn.ReLU())
    
    def forward(self,z,y):

        input = torch.cat([z,self.embed(y)],dim=1)
        input = self.fc1_1(input)
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        input = self.fc4_1(input)
        return input

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_1 = self.linear(4 ,32)
        self.fc2_1 = self.linear(32,32)
        self.fc3_1 = nn.Linear(32,1) ## f(x|y)

        self.embed = nn.Embedding(2,2)
    
    def linear(self,i,o):
        return nn.Sequential(nn.Linear(i,o),
               nn.LeakyReLU(0.2))
    
    def forward(self,x,y):

        input = torch.cat([x,self.embed(y)],dim=1)
        input = self.fc1_1(input)
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        return input

## Toy dataset samples comprises of two
## concentric circles. The  inner  blob
## refers to class '0' and outer circle
## is class '1' 

class ToyDataset(Dataset):
    def __init__(self,sample,noise,factor):
        super().__init__()

        X,y = make_circles(n_samples=sample,noise=noise,factor=factor)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]

def gradient_penalty(critic,real,fake,label,device="cpu"):
    B,N = real.shape
    alpha = torch.rand(B,1).repeat(1,N)
    alpha = alpha.to(device)

    interpolated_image = (real*alpha) +  (fake*(1-alpha))
    interpolated_score = critic(interpolated_image,label)

    gradients = torch.autograd.grad(
        outputs=interpolated_score,
        inputs =interpolated_image,
        grad_outputs =torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True)[0]
    
    gradients = gradients.view(gradients.shape[0],-1)
    gradients_norm = gradients.norm(2,dim=1)
    return torch.mean((gradients_norm-1)**2)

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Linear,nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

## WGAN-GP models
Cnet = Critic()
Gnet = Generator()

Cnet = Cnet.to(DEVICE)
Gnet = Gnet.to(DEVICE)

Coptim = Adam(Cnet.parameters(),LR,betas=(0.5,0.999))
Goptim = Adam(Gnet.parameters(),LR,betas=(0.5,0.999))

datasets = ToyDataset(1000,0.15,0.3)
dataload = DataLoader(datasets,BATCH_SIZE,shuffle=True)

for epoch in range(EPOCHS):

    for image, label  in  tqdm(dataload):
        image = image.float().to(DEVICE)
        label = label.long ().to(DEVICE)

        ## TRAINING
        Cnet.train()
        Gnet.train()

        for _ in range(CRITIC_ITERS):
            Coptim.zero_grad()

            noise = torch.randn(image.shape[0],NOISE)
            noise = noise.to(DEVICE)

            fakes = Gnet(noise,label)

            fake_logit = Cnet(fakes,label)
            real_logit = Cnet(image,label)
            penalty = gradient_penalty(Cnet,image,fakes,label,DEVICE)
            
            Closs = -(
                real_logit.mean() - 
                fake_logit.mean()) +\
                LAMBDA_GP*penalty
            
            Closs.backward(retain_graph=True)
            Coptim.step()
        
        Goptim.zero_grad()
        
        fake_logit = Cnet(fakes,label)
        Gloss = -fake_logit.mean()-IMP*torch.cdist(fakes,fakes).mean()
        Gloss.backward()
        Goptim.step()
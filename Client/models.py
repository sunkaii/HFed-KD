import torch
from torch import nn
import torch.nn.functional as F
    
class Generator(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(Generator,self).__init__()
        if dataset == 'cifar10':
            self.channel = 3
        elif dataset == 'fashionmnist':
            self.channel = 1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100,512,4,1,0,bias = False),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias = False),nn.BatchNorm2d(256),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias = False),nn.BatchNorm2d(128),nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias = False),nn.BatchNorm2d(64),nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64,self.channel,4,2,1,bias = False), nn.Tanh())
        self.embedding = nn.Embedding(10,100)
    def forward(self,noise,label):
        label_embedding = self.embedding(label)
        x = torch.mul(noise,label_embedding)
        x = x.view(-1,100,1,1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class CustomCNN_S(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(CustomCNN_S, self).__init__()
        if dataset == 'cifar10':
            self.input_channels = 3
            self.fc1_input_dim = 768 * 2 * 2
        elif dataset == 'fashionmnist':
            self.input_channels = 1
            self.fc1_input_dim = 768 * 1 * 1
        self.conv1 = nn.Conv2d(self.input_channels, 96, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 768, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.fc1_input_dim, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.fc3 = nn.Linear(768, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomCNN_M(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(CustomCNN_M, self).__init__()
        
        if dataset == 'cifar10':
            self.input_channels = 3
            self.fc1_input_dim = 1024 * 2 * 2
        elif dataset == 'fashionmnist':
            self.input_channels = 1
            self.fc1_input_dim = 1024 * 1 * 1
            
        self.conv1 = nn.Conv2d(self.input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.fc1_input_dim, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CustomCNN_L(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(CustomCNN_L, self).__init__()
        if dataset == 'cifar10':
            self.input_channels = 3
            self.fc1_input_dim = 1536 * 2 * 2
        elif dataset == 'fashionmnist':
            self.input_channels = 1
            self.fc1_input_dim = 1536 * 1 * 1
        
        self.conv1 = nn.Conv2d(self.input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(1024, 1536, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(self.fc1_input_dim, 3072)
        self.fc2 = nn.Linear(3072, 1536)
        self.fc3 = nn.Linear(1536, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CustomCNN_XS(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(CustomCNN_XS, self).__init__()
        
        if dataset == 'cifar10':
            self.input_channels = 3
            self.fc1_input_dim = 512 * 2 * 2
        elif dataset == 'fashionmnist':
            self.input_channels = 1
            self.fc1_input_dim = 512 * 1 * 1
            
        self.conv1 = nn.Conv2d(self.input_channels, 96, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.fc1_input_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomCNN_XL(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(CustomCNN_XL, self).__init__()
        
        if dataset == 'cifar10':
            self.input_channels = 3
            self.fc1_input_dim = 2048 * 2 * 2
        elif dataset == 'fashionmnist':
            self.input_channels = 1
            self.fc1_input_dim = 2048 * 1 * 1
            
        self.conv1 = nn.Conv2d(self.input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.fc1_input_dim, 4096) 
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
import torch
import torch.nn as nn
import numpy as np
import pdb

def sample(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
     # TODO: why normal and not uniform?
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q.astype('float32')

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight = sample((m.in_features, m.out_features)).from_numpy()
        m.weight.data.copy_(weight)
        nn.init.zeros_(m.bias)

class Discriminator1(nn.Module):
    def __init__(self, dim_latent=128, bn_flag=True):
        super(Discriminator1, self).__init__()
        self.dim_latent = dim_latent
        self.linear1 = nn.Linear(dim_latent, 1024)
        if self.bn_flag:
            self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.LeakyReLU(0.2)
        
        self.linear2 = nn.Linear(1024, 512)
        if self.bn_flag:
            self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.LeakyReLU(0.2)

        self.linear3 = nn.Linear(512, 256)
        if self.bn_flag:
            self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU(0.2)

        self.linear4 = nn.Linear(256, 256)
        if self.bn_flag:
            self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.LeakyReLU(0.2)

        self.linear5 = nn.Linear(256, 1)

    def forward(self, x):
        noise = (torch.randn(x.size()) * 0.3).cuda()
        x = self.linear1(x + noise)
        if self.bn_flag:
            x = self.bn1(x)
        x = self.relu1(x)
        noise = (torch.randn(x.size()) * 0.5).cuda()
        x = self.linear2(x + noise)
        if self.bn_flag:
            x = self.bn2(x)
        x = self.relu2(x)
        noise = (torch.randn(x.size()) * 0.5).cuda()
        x = self.linear3(x + noise)
        if self.bn_flag:
            x = self.bn3(x)
        x = self.relu3(x)
        noise = (torch.randn(x.size()) * 0.5).cuda()
        x = self.linear4(x + noise)
        if self.bn_flag:
            x = self.bn4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        return x.view(-1)

class Generator(nn.Module):
    def __init__(self, dim_latent=128, dim=64, bn_flag=True):
        super(Generator, self).__init__()
        self.dim_latent = dim_latent
        self.bn_flag = bn_flag
        self.dim = dim
        self.linear1 = nn.Linear(dim_latent, 4*4*4*dim)
        if bn_flag:
            self.bn1 = nn.BatchNorm1d(4*4*4*dim)
        self.relu1 = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(4*dim, 2*dim, 5, 2, 2, 1)
        if bn_flag:
            self.bn2 = nn.BatchNorm2d(2*dim)
        self.relu2 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(2*dim, dim, 5, 2, 2, 1)
        if bn_flag:
            self.bn3 = nn.BatchNorm2d(dim)
        self.relu3 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(dim, 3, 5, 2, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.dim_latent)
        x = self.linear1(x)
        if self.bn_flag:
            x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(-1, 4*self.dim, 4, 4)
        x = self.deconv1(x)
        if self.bn_flag:
            x = self.bn2(x)
        x = self.relu2(x)
        x = x.view(-1, 2*self.dim, 8, 8)
        x = self.deconv2(x)
        if self.bn_flag:
            x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(-1, self.dim, 16, 16)
        x = self.deconv3(x)
        x = self.tanh(x)
        return x, None, None

class Extractor(nn.Module):
    def __init__(self, dim_latent=128, dim=64, bn_flag=True, type_q='learn_std', std=0.1):
        super(Extractor, self).__init__()
        self.std = std
        self.type_q = type_q
        self.dim_latent = dim_latent
        self.bn_flag = bn_flag
        self.dim = dim
        self.pad1 = nn.ZeroPad2d((1, 2, 1, 2))
        self.conv1 = nn.Conv2d(3, dim, 5, 2)
        self.relu1 = nn.LeakyReLU(0.2)

        self.pad2 = nn.ZeroPad2d((1, 2, 1, 2))
        self.conv2 = nn.Conv2d(dim, dim*2, 5, 2)
        if bn_flag:
            self.bn1 = nn.BatchNorm2d(2*dim)
        self.relu2 = nn.LeakyReLU(0.2)

        self.pad3 = nn.ZeroPad2d((1, 2, 1, 2))
        self.conv3 = nn.Conv2d(dim*2, dim*4, 5, 2)
        if bn_flag:
            self.bn2 = nn.BatchNorm2d(4*dim)
        self.relu3 = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(4*4*4*dim, dim_latent)
        self.linear_std = nn.Linear(4*4*4*dim, dim_latent)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        if self.bn_flag:
            x = self.bn1(x)
        x = self.relu2(x)
        x = self.pad3(x)
        x = self.conv3(x)
        if self.bn_flag:
            x = self.bn2(x)
        x = self.relu3(x)
        x = x.view(-1, 4*4*4*self.dim)
        
        if self.type_q == 'learn_std':
            std = self.linear_std(x)
            std = torch.exp(std)
        elif self.type_q == 'fix_std':
            std = (torch.ones([x.size(0), self.dim_latent]) * self.std).cuda()
        else:
            std = None
            mean = None
        x = self.linear(x)

        if self.type_q in ['learn_std', 'fix_std']:
            epsilon = torch.randn([x.size(0), self.dim_latent]).cuda()
            mean = x
            std = mean + std * epsilon

        return x, mean, std

class Discriminator(nn.Module):
    def __init__(self, dim_latent=128, dim=64, bn_flag=True, dr_rate=0.2):
        super(Discriminator, self).__init__()
        self.dim=dim
        self.pad1 = nn.ZeroPad2d((1,2,1,2))
        self.conv1 = nn.Conv2d(3, dim, 5, 2)
        self.relu1 = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout(dr_rate)
        self.pad2 = nn.ZeroPad2d((1,2,1,2))
        self.conv2 = nn.Conv2d(dim, 2*dim, 5, 2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.drop2 = nn.Dropout(dr_rate)
        self.pad3 = nn.ZeroPad2d((1,2,1,2))
        self.conv3 = nn.Conv2d(2*dim, 4*dim, 5, 2)
        self.relu3 = nn.LeakyReLU(0.2)
        self.drop3 = nn.Dropout(dr_rate)

        self.linear1 = nn.Linear(dim_latent, 512)
        self.relu4 = nn.LeakyReLU(0.2)
        self.drop4 = nn.Dropout(dr_rate)

        self.linear2 = nn.Linear(512+4*4*4*dim, 512)
        self.relu5 = nn.LeakyReLU(0.2)
        self.drop5 = nn.Dropout(dr_rate)

        self.linear3 = nn.Linear(512, 1)

    def forward(self, x, z):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        z = self.drop4(self.relu4(self.linear1(z)))
        x = x.view(-1, 4*4*4*self.dim)
        x = torch.cat((x, z), dim=1)
        x = self.linear2(x)
        x = self.relu5(x)
        x = self.drop5(x)
        x = self.linear3(x)
        return x

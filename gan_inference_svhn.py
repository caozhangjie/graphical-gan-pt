from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net_svhn import Extractor, Discriminator, Generator, Discriminator1, init_weights
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pdb
import os
import plot
import save_images
import time

def generate_image(frame, fixed_noise, model, method, outf):
    x, x_mean, x_std = model(fixed_noise)
    samples = ((x.cpu()+1.0)*255.0/2.0).detach().numpy().astype('int32')
    save_images.save_images(
        samples.reshape((-1, 3, 32, 32)), 
        os.path.join(outf, '{}/{:06d}_samples.png'.format(method, frame))
    )

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_interval = 5000
print_interval = 100
batch_size = 64
std = .1
method = 'ali'
if method in ['vegan-kl', 'vegan-ikl', 'vegan-jsd']:
    type_q = 'learn_std' # learn_std, fix_std, no_std
    type_p = 'no_std'
    z_samples = 100 # MC estimation for D(q(z)||p(z))
elif method == 'vae':
    tpye_q = 'learn_std'
    type_p = 'learn_std'
else:
    type_q = 'no_std'
    type_p = 'no_std'
d_list = ['alice', 'alice-z', 'alice-x', 'vegan', 'vegan-wgan-gp', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vegan-mmd']
if method in d_list:
    distance_x = 'l2' # l1, l2
lambda_ = 1. # Balance reconstruction and regularization in vegan

dim = 64 # Model dimensionality
output_dim = 3072 # Number of pixels in svhn (3*32*32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
train_dset = SVHN('./data/svhn', 'train', download=True, transform=transform)
test_dset = SVHN('./data/svhn', 'test', download=True, transform=transform)
train_loader = DataLoader(train_dset, num_workers=4, pin_memory=True, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dset, num_workers=4, pin_memory=True, batch_size=1, shuffle=False)

num_iter = 200000
dim_latent = 128
lr=2e-4

os.system('mkdir -p ./log/'+method)
outf = './log/'
logfile = './log/'+method+'/log.txt'
open(logfile, 'w')

if method in ['vae']:
    beta1 = 0.9
else:
    beta1=0.5

if method in ['vegan', 'vegan-wgan-gp', 'vegan-kl', 'vegan-jsd', 'vegan-ikl']:
    bn_falg = False # Use batch_norm or not
    dim_latent = 8 # Dimensionality of the latent z
else:
    bn_flag = False
    dim_latent = 128
n_vis = batch_size*2 # Number of samples to be visualized

if method in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
    critic_iters = 0 # No discriminators
elif method in ['vegan', 'vegan-wgan-gp', 'wali', 'wali-gp']:
    critic_iters = 5 # 5 iters of D per iter of G
else:
    critic_iters = 1

extractor = Extractor(type_q=type_q)
extractor.apply(init_weights)
extarctor = extractor.cuda()
if method in ['vegan', 'vegan-wgan-gp']:
    discriminator = Discriminator1()
elif method in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
    pass
else:
    discriminator = Discriminator()
discriminator.apply(init_weights)
discriminator = discriminator.cuda()

generator = Generator()
generator.apply(init_weights)
generator = generator.cuda()

criterion = nn.BCEWithLogitsLoss()


optimizer_e = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))


if method in ['ali','vae','wali', 'wali-gp']:
    rec_penalty = None
elif method == 'alice-z':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
elif method == 'alice-x':
    rec_penalty = 1.*lib.utils.distance.distance(p_z, rec_z, DISTANCE_X)
elif method == 'alice':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
    rec_penalty += 1.*lib.utils.distance.distance(p_z, rec_z, DISTANCE_X)
elif method == 'vegan':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
elif method == 'vegan-wgan-gp':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
elif method == 'vegan-mmd':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
elif method == 'vegan-kl':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
elif method == 'vegan-ikl':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
elif method == 'vegan-jsd':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
else:
    raise('NotImplementedError')

fixed_noise = torch.randn([n_vis, dim_latent]).cuda()
len_loader = len(train_loader)

for iter_ in range(num_iter):
    start_time = time.time()
    for sub_iter in range(critic_iters + 1):
        true_iter = iter_ * (critic_iters + 1) + sub_iter
        if true_iter % len_loader == 0:
            iter_loader = iter(train_loader)
        data = iter_loader.next()
        real_x, label = data
        real_x = real_x.cuda()
        q_z, q_z_mean, q_z_std = extractor(real_x)
        rec_x, rex_x_mean, rec_x_std = generator(q_z)
        p_z = torch.randn([input_.size(0), dim_latent]).cuda()
        fake_x, _, _ = generator(p_z)
        rec_z, _, _ = extractor(fake_x)
        if method in ['vegan-kl', 'vegan-ikl', 'vegan-jsd']:
            p_z_mean = torch.zeros([z_samples, dim_latent]).float().cuda()
            p_z_std = torch.ones([z_samples, dim_latent]).float().cuda()
        elif method == 'vae':
            p_z_mean = torch.zeros([64, dim_latent]).float().cuda()
            p_z_std = torch.ones([64, dim_latent]).float().cuda()
        if method in ['vegan', 'vegan-wgan-gp']:
            disc_real = discriminator(p_z)
            disc_fake = discriminator(q_z)
        elif method in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
            pass
        else:
            disc_real = discriminator(real_x, q_z)
            disc_fake = discriminator(fake_x, p_z)
        if method == 'ali':
            label_dis = torch.cat((torch.zeros([input_.size(0)]), torch.ones([input_.size(0)])), dim=0).cuda()
            label_gen = torch.cat((torch.ones([input_.size(0)]), torch.zeros([input_.size(0)])), dim=0).cuda()
            disc_input = torch.cat((disc_real, disc_fake), dim=0)
            if sub_iter == 0:
                disc_loss = criterion(disc_input, label_gen)
                optimizer_g.zero_grad()
                optimizer_e.zero_grad()
                disc_loss.backward()
                optimizer_g.step()
                optimizer_e.step()
            else:
                disc_loss = criterion(disc_input, label_dis)
                optimizer_d.zero_grad()
                disc_loss.backward()
                optimizer_d.step()
    if method in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
        if iter_ > 0:
            plot.plot('train gen cost ', gen_loss.cpu().item())
    else:
        plot.plot('train disc cost', disc_loss.cpu().item())
    plot.plot('time', time.time() - start_time)
    '''
    if iter_ % 100 == 99:
        if rec_penalty is not None:
            dev_rec_costs = []
            dev_reg_costs = []
            for images,_ in dev_gen():
                _dev_rec_cost, _dev_gen_cost = session.run(
                    [rec_penalty, gen_cost], 
                    feed_dict={real_x_int: images}
                )
                dev_rec_costs.append(_dev_rec_cost)
                dev_reg_costs.append(_dev_gen_cost - _dev_rec_cost)
            plot.plot('dev rec cost', np.mean(dev_rec_costs))
            plot.plot('dev reg cost', np.mean(dev_reg_costs))
        else:
            dev_gen_costs = []
            for images,_ in dev_gen():
                _dev_gen_cost = session.run(
                    gen_cost, 
                    feed_dict={real_x_int: images}
                )
                dev_gen_costs.append(_dev_gen_cost)
            plot.plot('dev gen cost', np.mean(dev_gen_costs))
    '''
    # Write logs
    if (iter_ < 5) or (iter_ % print_interval == print_interval-1):
        plot.flush(os.path.join(outf, method), logfile)
    plot.tick()

    # Generation and reconstruction
    if iter_ % test_interval == test_interval-1:
        generator.eval()
        discriminator.eval()
        extractor.eval()
        generate_image(iter_, fixed_noise, generator, method, outf)
        ori_images = []
        gen_images = []
        for batch_id, data in enumerate(test_loader):
            input_, label = data
            input_ = input_.cuda()
            sample, sample_mean, sample_std = extractor(input_)
            rec_x, rex_x_mean, rec_x_std = generator(sample)
            ori_images.append(((input_.cpu()+1.0)*255.0/2).detach().numpy().astype('int32'))
            gen_images.append(((rec_x.cpu()+1.0)*255.0/2).detach().numpy().astype('int32'))
        ori_images = np.array(ori_images)
        gen_images = np.array(gen_images)
        save_images.save_images(
            ori_images.reshape((-1, 3, 32, 32)), 
            os.path.join(outf, '{}/{:06d}_origin.png'.format(method, iter_))
        )
        save_images.save_images(
            gen_images.reshape((-1, 3, 32, 32)), 
            os.path.join(outf, '{}/{:06d}_gen.png'.format(method, iter_))
        )
        generator.train()
        discriminator.train()
        extractor.train()

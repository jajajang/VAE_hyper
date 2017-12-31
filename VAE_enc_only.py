import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

dir_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = (not args.no_cuda) and torch.cuda.is_available()



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
EPS=1e-5

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1)

    def forward(self, x):
        mu= self.encode(x.view(-1, 784))
        return mu


enc_ = encoder()

if args.cuda:
    enc_.cuda()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 784

    return BCE + KLD
'''
def proj(params):
    paramsy=params.clone()
    for i in range(args.batch_size):
	    paramsy[i]=params[i]/(params[i].norm()).clamp(min=1-EPS)*(1-EPS)
    return paramsy
'''


def proj(params):
    paramsy=params.clone()
    t_val = (params.norm(p=2, dim=1)**2+1).sqrt()
    for i in range(args.batch_size):
        paramsy[i] = params[i]/(1+t_val[i])
    return paramsy


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


"""
def distance(u, v):
    uu = u.norm() ** 2
    vv = v.norm() ** 2
    uv = torch.dot(u,v)
    alpha = 1 - uu
    alpha = alpha.clamp(min=EPS)
    beta = 1 - vv
    beta = beta.clamp(min=EPS)

    gamma = 1 + 2 * (uu - 2 * uv + vv) / (alpha * beta)
    gamma = gamma.clamp(min=1 + EPS)

    return arcosh(gamma)
"""


def distance(u,v):
    uu = torch.dot(u, u)
    vv = torch.dot(v, v)
    u0 = (uu+1)
    v0 = (vv+1)
    d = arcosh(u0.sqrt()*v0.sqrt()-torch.dot(u,v))
    return d


optimizer_enc = optim.Adam(enc_.parameters(), lr=1e-3)


mew=0
labby=0
grady=0



def train(epoch):
    global mew
    global labby
    global grady
    enc_.train()
    train_loss = 0
    prev_data=0
    skipit=0
    for batch_idx, (data, label) in enumerate(train_loader):
        go_skip=False
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer_enc.zero_grad()
        mu = enc_(data)
        loss = punisher(mu,label)
        loss.backward()
        for i, ass in enumerate(enc_.parameters()):
            if ass is None:
                continue
            elif np.isnan(((ass.grad).data).numpy()).any():
                print mu
                mew=mu
                labby=label
                go_skip=True

        if (not go_skip):
            train_loss += loss.data[0]
            optimizer_enc.step()
        else:
            optimizer_enc.zero_grad()
            skipit+=1
        grady=ass.grad.data
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            print 'Skip number : '+str(skipit)
            skipit=0

    print '====> Epoch: '+ str(epoch)+' Average loss: '+ str(
	  train_loss / len(train_loader.dataset))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def punisher(z, label):
    same_family=0
    diff_family=0
    for i, latent_1 in enumerate(z):
        for j, latent_2 in enumerate(z):
            if i>=j: continue
            elif label[i] == label[j]:
                same_family += torch.exp(-distance(latent_1, latent_2))
            else:
                diff_family += torch.exp(-distance(latent_1, latent_2))

    return -torch.log(same_family)+torch.log(diff_family)


def test(epoch):
    enc_.eval()
    test_loss = 0
    for i, (data, label) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        mu= enc_(data)
        test_loss += 0.1*punisher(mu,label).data[0]
        if i == 0:
          n = min(data.size(0), 8)
    test_loss /= len(test_loader.dataset)

    print '<-------------TEST LOSS------------->'
    print test_loss


def plot(filename):
    enc_.eval()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0, 0), 1., color='black', fill=False)
    ax.add_artist(circle)

    for i, (data, label) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        mu = enc_(data)
        mu_disk=proj(mu)
        for j, z in enumerate(mu_disk):
    	    if label[j]==0:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C0')
    	    elif label[j]==1:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C1')
    	    elif label[j]==2:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C2')
    	    elif label[j]==3:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C3')
    	    elif label[j]==4:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C4')
    	    elif label[j]==5:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C5')
    	    elif label[j]==6:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C6')
    	    elif label[j]==7:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C7')
    	    elif label[j]==8:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C8')
    	    elif label[j]==9:
        	    	ax.plot(z.data[0],z.data[1], 'o', color='C9')

    plt.savefig(filename)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    plot('Latent_0.1_mu_only_'+str(epoch)+'epoch.png')



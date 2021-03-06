import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = (not args.no_cuda) and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
EPS = 1e-5

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 2)
        self.fc22 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        return self.reparameterize(mu, logvar)


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        return self.decode(z)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()
model.load_state_dict('Discriminator.pt')

enc_ = encoder()
dec_ = decoder()

if args.cuda:
    enc_.cuda()
    dec_.cuda()


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


def proj(params):
    paramsy = params.clone()
    t_val = (params.norm(p=2, dim=1) ** 2 + 1).sqrt()
    for i in range(args.batch_size):
        paramsy[i] = params[i] / (1 + t_val[i])
    return paramsy


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))

'''
def distance(u, v):
    uu = u.norm() ** 2
    vv = v.norm() ** 2
    u0 = (uu + 1)
    v0 = (vv + 1)
    d = arcosh(u0.sqrt() * v0.sqrt() - torch.dot(u, v))
    return d.clamp(min=EPS)
'''

def distance(u, v):
    uu = u.norm() ** 2
    vv = v.norm() ** 2
    u0 = (uu + 1)
    v0 = (vv + 1)
    d = arcosh(u0.sqrt() * v0.sqrt() - torch.dot(u, v))
    return d.clamp(min=EPS)

def punisher(z, label):
    same_family = 0
    diff_family = 0
    for i, latent_1 in enumerate(z):
        for j, latent_2 in enumerate(z):
            if i >= j:
                continue
            elif label[i] == label[j]:
                same_family += torch.exp(-distance(latent_1, latent_2))
            else:
                diff_family += torch.exp(-distance(latent_1, latent_2))

    return -torch.log(same_family) + torch.log(diff_family)


optimizer_enc = optim.Adam(enc_.parameters(), lr=1e-3)
optimizer_dec = optim.Adam(dec_.parameters(), lr=1e-3)


def train(epoch):
    enc_.train()
    dec_.train()
    train_loss = 0
    skipit=0
    skiptotal=0
    VAELoss=0
    for batch_idx, (data, label) in enumerate(train_loader):
        go_skip=False
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        z, mu, logvar = enc_(data)
        recon_batch = dec_(z)
        VLoss = F.binary_cross_entropy(recon_batch, data.view(-1, 784))
        loss = loss_function(recon_batch, data, mu, logvar) + 0.1 * punisher(z, label)
        loss.backward()

        for i, ass in enumerate(enc_.parameters()):
            if ass.grad is None:
                continue
            elif np.isnan(((ass.grad).data.cpu()).numpy()).any():
                go_skip=True

        if (not go_skip):
            train_loss += loss.data[0]
            VAELoss += VLoss.data[0]
            optimizer_enc.step()
            optimizer_dec.step()
        else:
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            skipit+=1
            skiptotal+=1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            print 'Skip number : '+str(skipit)
            skipit=0

    print '====> Epoch: ' + str(epoch) + ' Average loss: ' + str(
        train_loss / (len(train_loader.dataset)-skiptotal))
    print '====> Epoch: ' + str(epoch) + ' Average loss: ' + str(
        VAELoss / (len(train_loader.dataset)-skiptotal))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))



def test(epoch):
    enc_.eval()
    dec_.eval()
    correct=0
    total=0
    for i, (data, label) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        z, mu, logvar = enc_(data)
        recon_batch = dec_(z)
        sampy=(recon_batch.size())[0]
        new_label=model(recon_batch.view(list(recon_batch.size())[0], 1, 28, 28)).data.cpu().numpy()
        for i in range(sampy):
            total+=1
            if label[i]==np.argmax(new_label[i]):
                correct+=1
    print '<-------------TEST LOSS------------->'
    print correct*(1.0)/total


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
        zzzz,mu,logvar = enc_(data)
        mu_disk = proj(mu)
        for j, z in enumerate(mu_disk):
            if label[j] == 0:
                ax.plot(z.data[0], z.data[1], 'o', color='C0')
            elif label[j] == 1:
                ax.plot(z.data[0], z.data[1], 'o', color='C1')
            elif label[j] == 2:
                ax.plot(z.data[0], z.data[1], 'o', color='C2')
            elif label[j] == 3:
                ax.plot(z.data[0], z.data[1], 'o', color='C3')
            elif label[j] == 4:
                ax.plot(z.data[0], z.data[1], 'o', color='C4')
            elif label[j] == 5:
                ax.plot(z.data[0], z.data[1], 'o', color='C5')
            elif label[j] == 6:
                ax.plot(z.data[0], z.data[1], 'o', color='C6')
            elif label[j] == 7:
                ax.plot(z.data[0], z.data[1], 'o', color='C7')
            elif label[j] == 8:
                ax.plot(z.data[0], z.data[1], 'o', color='C8')
            elif label[j] == 9:
                ax.plot(z.data[0], z.data[1], 'o', color='C9')

    plt.savefig(filename)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

torch.save(enc_.state_dict(), 'HypD2beta01dec.pt')
torch.save(dec_.state_dict(), 'HypD2beta01dec.pt')

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


enc_ = encoder()
dec_ = decoder()
model = Net()
if args.cuda:
    model.cuda()

model.load_state_dict(torch.load('Discriminator.pt'))

if args.cuda:
    enc_.cuda()
    dec_.cuda()

enc_.load_state_dict(torch.load('enc_training_not_Poin_dec01.pt'))
dec_.load_state_dict(torch.load('dec_training_not_Poin_dec01.pt'))


def test():
    enc_.eval()
    dec_.eval()
    correct=0
    total=0
    wrong=0
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
            else:
                wrong+=1
                if wrong%100==0:
                    comparison = torch.cat([data[i],
                                            recon_batch.view(args.batch_size, 1, 28, 28)[i]])
                    save_image(comparison.data.cpu(),
                       'results_wrong/beep_' + str(wrong) +'_label_'+str(label[i])
                               +str(np.argmax(new_label[i]))+ '.png', nrow=2)
    print '<-------------TEST LOSS------------->'
    print correct*(1.0)/total


test()

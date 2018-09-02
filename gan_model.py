import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import misc

# Datasets
trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                      download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.MNIST(root="./data", train=False,
                                     download=True, transform=transforms.ToTensor())

# Loaders for Datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)


class Generator(nn.Module):
    def __init__(self):
        """
        Init for the Generator Class, defines the layers of the network
        """
        super(Generator, self).__init__()

        self.conv1 = nn.ConvTranspose2d(100, 128, kernel_size=7, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(128, 56, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(56, 28, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(28, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Function that defines a forward pass through the network and the computations that take place
        :param x: Generated noise
        :return: Matrix of [-1, 1, 28, 28]
        """
        x = x.view(-1, 100, 1, 1)
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        x = torch.tanh(x)
        return x


def generate_img():
    """
    Simple function that generates a randomized vector of 100 values between [-1,1]
    and passes it into the generator.
    :return: A single generated image
    """
    noise = torch.randn(1, 100)
    gen = Generator()
    image = gen(noise)
    image = image.view(28, 28)
    image = image.detach()
    misc.toimage(image).show()
    return image


generate_img()


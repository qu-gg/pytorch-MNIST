import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import misc
import numpy as np
import random

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

        self.conv1 = nn.ConvTranspose2d(100, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 56, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(56, 28, kernel_size=4, stride=2)
        self.conv4 = nn.ConvTranspose2d(28, 1, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Function that defines a forward pass through the network and the computations that take place
        :param x: Generated noise
        :return: Matrix of [-1, 1, 28, 28]
        """
        x = x.view(-1, 100, 1, 1)
        x = self.bn1(self.conv1(x))
        x = self.conv2(x)
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        x = torch.tanh(x)
        return x


def generate_img(gen, noise, show=False):
    """
    Simple function that generates a randomized vector of 100 values between [-1,1]
    and passes it into the generator for a specified number of images.
    :return: A batch of generated images
    """
    image = gen(noise)
    if show:
        reshaped = image.view(28, 28).detach()
        return reshaped
    return image


def get_batch(gen, num):
    """
    Function to generate a batch of fake images
    :param gen: Generator to use
    :param num: Number of images in batch
    :return: Tensor of generated images
    """
    classes = []
    for _ in range(num):
        classes.append(random.uniform(0.7, 1.0))

    images = []
    for _ in range(num):
        noise = torch.randn(1, 100)
        img = generate_img(gen, noise)
        images.append(img)

    return torch.cat(images), torch.Tensor(classes).long()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=6, stride=2)
        self.conv2 = nn.Conv2d(128, 56, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(56, 100, kernel_size=4, stride=2)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = x.view(-1, 100)
        x = torch.sigmoid(self.fc(x))
        return x


gen = Generator()
discrim = Discriminator()

# Loss
cross_entropy = nn.CrossEntropyLoss()

# Optimizer
dis_optim = optim.Adam(discrim.parameters(), lr=.001)
gen_optim = optim.Adam(gen.parameters(), lr=.001)


def training(num_epochs):
    """
    Function to handle the training of the GAN
    :return: None
    """
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            # Zero grad
            dis_optim.zero_grad()
            gen_optim.zero_grad()

            # Prob for Fake Images
            gen_batch, gen_labels = get_batch(gen, 64)
            dis_fake = discrim(gen_batch)

            # Prob for Real Images
            images, _ = data
            labels = torch.Tensor([random.uniform(0.0, 0.3) for _ in range(64)]).long()
            dis_real = discrim(images)

            # Loss
            dis_loss = cross_entropy(dis_real, labels)
            gen_loss = cross_entropy(dis_fake, gen_labels)

            dis_loss.backward()
            gen_loss.backward()

            # Optimizer step
            dis_optim.step()
            gen_optim.step()

            if i % 10 == 0:
                print("Discrim Loss on ", i, ": ", dis_loss.detach().numpy())
                print("Gen Loss on ", i, ": ", gen_loss.detach().numpy())
                noise = torch.randn(1, 100)
                image = generate_img(gen, noise, True)
                misc.imsave("results/" + str(i) + ".jpg", image)

training(1)
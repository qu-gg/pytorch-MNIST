import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import imageio
import numpy as np

# Datasets
trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                      download=True, transform=transforms.ToTensor())

# Loaders for Datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


def mnist_batch(num_wanted, batch_size):
    """
    Function to generate a batch of mnist pictures that are of a specific number
    This is to help train the GAN in a more focused way
    :param num_wanted: The digit to get images of
    :param batch_size: How many images in a batch
    :return: Batch of MNIST images of a specific digit
    """
    images = []
    while len(images) < batch_size:
        mnist, labels = next(iter(trainloader))
        for i in range(len(labels)):
            if len(images) >= batch_size:
                break
            if labels[i] == num_wanted:
                images.append(mnist[i])
    tensor = torch.cat(images)
    tensor = tensor.view(batch_size, 1, 28, 28)
    return tensor


class Generator(nn.Module):
    def __init__(self):
        """
        Init for the Generator Class, defines the layers of the network
        """
        super(Generator, self).__init__()

        self.conv1 = nn.ConvTranspose2d(100, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 56, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(56)
        self.conv3 = nn.ConvTranspose2d(56, 28, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(28)
        self.conv4 = nn.ConvTranspose2d(28, 1, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Function that defines a forward pass through the network and the computations that take place
        :param x: Generated noise
        :return: Matrix of [-1, 1, 28, 28]
        """
        x = x.view(-1, 100, 1, 1)
        x = self.bn1(self.conv1(x))
        x = f.leaky_relu(self.bn2(self.conv2(x)))
        x = f.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
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
    classes = [np.random.uniform(0.9, 1.0) for _ in range(num)]

    images = []
    for _ in range(num):
        noise = torch.randn(1, 100)
        img = generate_img(gen, noise)
        images.append(img)

    return torch.cat(images), torch.Tensor(classes)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=6, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 100, kernel_size=4, stride=2)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = x.view(-1, 100)
        x = torch.sigmoid(self.fc(x))
        return x


# Initialization for the Generator and Discriminator
gen = Generator()
discrim = Discriminator()

# Loss
cross_entropy = nn.BCELoss()

# Optimizer
dis_optim = optim.Adam(discrim.parameters(), lr=.00005)
gen_optim = optim.Adam(gen.parameters(), lr=.00005)


def training(num_epochs, num_steps):
    """
    Function to handle the training of the GAN
    :return: None
    """
    for epoch in range(num_epochs):
        print("\nEpoch: {}".format(epoch))
        for i in range(num_steps):
            # Zero grad
            dis_optim.zero_grad()

            # Train on real
            images = mnist_batch(9, 64)
            labels = torch.Tensor([np.random.uniform(0.0, 0.1) for _ in range(64)])
            dis_real = discrim(images)
            dis_real_loss = cross_entropy(dis_real, labels)
            dis_real_loss.backward()

            # Prob for Fake Images
            gen_batch, gen_labels = get_batch(gen, 64)
            dis_fake = discrim(gen_batch)
            dis_fake_loss = cross_entropy(dis_fake, gen_labels)
            dis_fake_loss.backward()

            dis_optim.step()

            if i % 10 == 0:
                print("Discrim loss on {}: {}".format(i, dis_fake_loss.detach().numpy()))

        for i in range(num_steps*2):
            # Zero grad
            gen_optim.zero_grad()

            gen_batch, gen_labels = get_batch(gen, 64)
            dis_pred = discrim(gen_batch)
            gen_loss = -cross_entropy(dis_pred, gen_labels)
            gen_loss.backward()
            gen_optim.step()

            if i % 10 == 0 or i + 1 == num_steps*2:
                print("Genera Loss on {}: {}".format(i, gen_loss.detach().numpy()))
                img = generate_img(gen, torch.randn(1, 100), True)
                for _ in range(10):
                    noise = torch.randn(1, 100)
                    image = generate_img(gen, noise, True)
                    img = np.concatenate((img, image), axis=1)
                imageio.imsave("output/{}epoch{}num.jpg".format(epoch, i), img)


train = int(input("Input num runs per epoch: "))
training(100, train)

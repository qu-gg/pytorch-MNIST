import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from scipy import misc

BATCH_SIZE = 64

# Dataset + Loader
trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                        download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)


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


class Encoder(nn.Module):
    """
    Encoder portion of the model. Takes in source vectors of [BATCH_SIZE, 784]
    and encodes them into a latent vector of length 50.

    This squishes the
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 150, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(150, 100,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(100, 50, kernel_size=5, stride=2)

    def forward(self, x):
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.expand1 = nn.ConvTranspose2d(50, 100, kernel_size=3, stride=2)
        self.expand2 = nn.ConvTranspose2d(100, 150, kernel_size=3, stride=2)
        self.expand3 = nn.ConvTranspose2d(150, 200, kernel_size=2, stride=2)
        self.expand4 = nn.ConvTranspose2d(200, 1, kernel_size=2, stride=2)

    def forward(self, x):
        x = f.leaky_relu(self.expand1(x))
        x = f.leaky_relu(self.expand2(x))
        x = f.leaky_relu(self.expand3(x))
        x = self.expand4(x)
        return x


# Models
encode = Encoder()
decode = Decoder()

# Optimizers + Loss
encode_opt = optim.Adam(encode.parameters(), lr=.0001)
decode_opt = optim.Adam(decode.parameters(), lr=.0001)

cross_entropy = nn.MSELoss()


def train(num_epochs, num_steps):
    for epoch in range(num_epochs):
        for i in range(num_steps):
            images = mnist_batch(7, BATCH_SIZE)
            encoded = encode(images)
            decoded = decode(encoded)

            loss = cross_entropy(decoded, images)
            loss.backward()

            encode_opt.step()
            decode_opt.step()

            if i == 99:
                print("Loss on epoch {} at step {}: {}".format(epoch, i, loss))
                misc.imsave("results/autoencode/{}{}original.jpg".format(epoch, i), images[0].view(28, 28))
                misc.imsave("results/autoencode/{}{}encoded.jpg".format(epoch, i), encoded[0].view(5, 10).detach().numpy())
                misc.imsave("results/autoencode/{}{}decoded.jpg".format(epoch, i), decoded[0].view(28, 28).detach().numpy())


def test_net(num_wanted):
    image = mnist_batch(num_wanted, 1)
    encoded = encode(image)
    decoded = decode(encoded)
    misc.imsave("results/autoencode/aaa-{}originaltest.jpg".format(num_wanted), image.view(28, 28))
    misc.imsave("results/autoencode/aaa-{}encoded.jpg".format(num_wanted), encoded.view(5, 10).detach().numpy())
    misc.imsave("results/autoencode/aaa-{}decoded.jpg".format(num_wanted), decoded.view(28, 28).detach().numpy())


train(10, 100)
test_net(3)
test_net(7)
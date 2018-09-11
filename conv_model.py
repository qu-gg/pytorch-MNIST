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


class Net(nn.Module):
    # Size of a batch being fed into the network
    batch_size = 64

    def __init__(self):
        """
        Initialization for the network, defining the hidden layers of the model
        """
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, self.batch_size)
        self.fc2 = nn.Linear(self.batch_size, 10)

    def forward(self, x):
        """
        Simple linear feed forward network with sigmoid as the activation functions on the
        first 3 layers. Outputs the raw guess for use with argmax
        :param x: input to the network
        :return: output of the network
        """
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = f.relu(self.fc1(x))
        output = self.fc2(x)
        return output


net = Net()                                         # Assigning the net to a local variable

cross_entropy = nn.CrossEntropyLoss()               # Standard Cross Entropy loss is a good loss function
optimizer = optim.Adam(net.parameters(), lr=.001)   # Adam optimizer works well with a lower learning rate


def train(num_epoch):
    """
    Function that handles the training loop of the network
    :param num_epoch: number of times to loop through the dataset
    :return: None
    """
    set_batch(64)
    for epoch in range(num_epoch):
        for i, data in enumerate(trainloader, 0):
            images, labels = data

            # Zero grad
            optimizer.zero_grad()

            # forward, backward, optimize
            output = net(images)
            loss = cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            # print results
            if i % 100 == 0:
                print("Loss at iter", i, "in epoch", epoch, ": ", loss.data)

    print("Finished training")


def test_net():
    """
    Function to handle testing the network for accuracy on the test set
    The test images are images the network has never seen before
    :return: None
    """
    total = 0
    correct = 0
    set_batch(64)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            print(images[0])
            pred = net(images)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the test set: ", 100 * correct / total, "%")
    save_net(net)


def save_net(model):
    """
    Function to save the model at any current stage
    :param model: model to save
    :return: None
    """
    print("Saving net...", end="")
    torch.save(model, "./model")
    print("...complete!")


def set_batch(num):
    """
    Function to set the batch size and layer sizes
    :param num: Batch size
    :return: None
    """
    setattr(net, "batch_size", num)


def use_net():
    """
    Function that takes an input file, preprocesses the data, and feeds it into the
    network for a prediction
    :return: None
    """
    net = torch.load("./model")
    with torch.no_grad():
        while True:
            # get image path
            path = input("Input image file (q to quit): ")
            if path == "q":
                break

            # create the image array and size it as a Torch tensor
            image = misc.imread(path)
            image_rgb_raw = [[] for _ in range(28)]

            for row_index in range(len(image)):
                for pixel_index in range(len(image[row_index])):
                    pixel_value = image[row_index][pixel_index][0]
                    if pixel_value == 255:
                        image_rgb_raw[row_index].append(0)
                    else:
                        image_rgb_raw[row_index].append(1 - (pixel_value) / 255)

            img = torch.as_tensor(image_rgb_raw)
            img = img.view(1, 1, 28, 28)

            # use the net
            set_batch(1)
            output = net(img)
            print(output)

            # print prediction
            pred = torch.argmax(output)
            print("Your digit is", pred)


train(5)

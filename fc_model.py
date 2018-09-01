import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Datasets
trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                      download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.MNIST(root="./data", train=False,
                                     download=True, transform=transforms.ToTensor())

# Loaders for Datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)


class Net(nn.Module):

    def __init__(self):
        """
        Initialization for the network, defining the hidden layers of the model
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 100)
        self.final = nn.Linear(100, 10)

    def forward(self, x):
        """
        Simple linear feed forwarc network with sigmoid as the activation functions on the
        first 3 layers. Outputs the raw guess for use with argmax
        :param x: input to the network
        :return: output of the network
        """
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        output = self.final(x)
        return output


net = Net()     # Assigning the net to a local variable

cross_entropy = nn.CrossEntropyLoss()   # Standard Cross Entropy loss is a good loss function
optimizer = optim.Adam(net.parameters(), lr=.001)   # Adam optimizer works well with a lower learning rate


def train(num_epoch):
    """
    Function that handles the training loop of the network
    :param num_epoch: number of times to loop through the dataset
    :return: None
    """
    for epoch in range(num_epoch):
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images = images.view(-1, 784)

            # Zero grad
            optimizer.zero_grad()

            # forward, backward, optimize
            output = net(images)
            loss = cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            # print results
            if i % 100 == 0:
                print("Loss at iter ", i, ": ", loss.data)

    print("Finished training")


def test_net():
    """
    Function to handle testing the network for accuracy on the test set
    The test images are images the network has never seen before
    :return: None
    """
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(-1, 784)

            pred = net(images)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the test set: ", 100 * correct / total, "%")


train(1)
test_net()

#C:\Users\skbla\Envs\py1\Scripts\python.exe

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data = "D:\\MLDATA\\dataset"

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '\\train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '\\test1', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# define a model from torch vision
# https://pytorch.org/docs/0.3.0/torchvision/models.html
model = models.densenet121(pretrained=True)

## TODO: Use a pretrained model to classify the cat and dog images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training model on device: {}...".format(device))

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(nn.Linear(1024, 500),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(500, 2),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

epochs = 1
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        model.train()
        # reset the gradients
        optimizer.zero_grad()
        # forward pass
        log_ps = model.forward(images)
        loss = criterion(log_ps,labels)
        # backpropagate (train)
        loss.backward()
        optimizer.step()
        running_loss += loss
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images,labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model.forward(images)
                loss = criterion(log_ps,labels)
                test_loss += loss
                ps = torch.exp(log_ps)
                top_p,top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
    model.train()
    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
          "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

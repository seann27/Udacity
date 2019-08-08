# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import pretrained_models
import time
import numpy as np
from PIL import Image
import argparse
import os
from CustomClassifiers import Classifier

# Load default options
save_dir = '.'
arch = 'vgg19'
model = pretrained_models.getPretrainedModel(arch)
learning_rate = 0.001
hidden_units = None
epochs = 10
filename = save_dir+'/'+arch+'_checkpoint.pth'

# Load commandline options
parser = argparse.ArgumentParser(description="Load model and hyperparameters")
parser.add_argument('--save_dir', help='Directory where model checkpoint is saved')
parser.add_argument('--filename', help='Name of checkpoint file')
parser.add_argument('--arch', help='Model to be used from torchvision')
parser.add_argument('--learning_rate', help='Learning rate for training classifier')
parser.add_argument('--hidden_units', metavar='N', type=int, nargs='+',
                    help='Hidden units used for training classifier')
parser.add_argument('--epochs', help='Number of training iterations')
args = parser.parse_args()

if args.save_dir:
    save_dir = args.save_dir
if args.arch:
    filename = save_dir+'/'+args.arch+'_checkpoint.pth'
    model = pretrained_models.getPretrainedModel(args.arch)
    arch = args.arch
if args.filename:
    filename = save_dir+'/'+filename
if args.learning_rate:
    learning_rate = args.learning_rate
if args.hidden_units:
    hidden_units = args.hidden_units
if args.epochs:
    epochs = int(args.epochs)

print('directory: ',save_dir)
print('model: ',arch)
print('learning rate: ',learning_rate)
print('hidden units: ',hidden_units)
print('epochs: ',epochs)
print('checkpoint file: ',filename)

if os.path.isfile(filename):
    os.remove(filename)

# Load the data images
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transform images
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_val_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform=test_val_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_val_transforms)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Configure device (cpu or gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training model on {} ...".format(device))

# Turn off grad for pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# set classifier
if hidden_units is None:
    classifier = nn.Sequential(nn.Linear(25088, 3072),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(3072, 1024),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(1024, 306),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(306, 102),
                               nn.LogSoftmax(dim=1))
else:
    classifier = Classifier(hidden_units)

model.classifier = classifier
print(model.classifier)

# Define loss function and optimizer for backpropagation
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

# Send the model to the device
model.to(device)

# train and validate model
total_start = time.time()
for e in range(epochs):
    running_loss = 0
    step = 1
    start = time.time()
    # train step
    for images, labels in trainloader:
        # send images and lables to the device
        images, labels = images.to(device), labels.to(device)

        # turn training mode on
        model.train()

        # reset gradients
        optimizer.zero_grad()

        # forward pass
        log_ps = model.forward(images)
        loss = criterion(log_ps,labels)

        # backpropagate
        loss.backward()
        optimizer.step()

        # increment loss
        running_loss += loss

        # track progress
        if step % 41 == 0 or step == 1:
            print("Epoch: {}/{} ... ".format(e+1,epochs),
                  "Training Step: {}/{} ... ".format(step,len(trainloader)),
                  "Time elapsed: {:.3f} seconds".format((time.time()-start)/3))
        step += 1

    else: # End training iteration, begin validation
        print("Epoch: {}/{} ... ".format(e+1,epochs), "Training complete.")
        print("Epoch: {}/{} ... ".format(e+1,epochs), "Validating... ")
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images,labels in validloader:
                # configure test data for device
                images,labels = images.to(device),labels.to(device)

                # forward pass
                log_ps = model.forward(images)
                loss = criterion(log_ps,labels)
                test_loss += loss

                # get accuracy
                ps = torch.exp(log_ps)
                top_p,top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

    # turn train mode back on
    model.train()

    # print progress
    print("Epoch: {}/{} ... ".format(e+1, epochs),
      "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
      "Test Accuracy: {:.3f}%".format(accuracy/len(validloader)*100))

# print total time taken for model to be trained
print("Total Time: {:.3f} seconds".format((time.time()-total_start)/3))

# test the network
accuracy = 0
with torch.no_grad():
    model.eval()
    for images,labels in testloader:
        # configure test data for device
        images,labels = images.to(device),labels.to(device)

        # forward pass
        log_ps = model.forward(images)
        loss = criterion(log_ps,labels)

        # get accuracy
        ps = torch.exp(log_ps)
        top_p,top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    # print results
    print("Test Accuracy: {:.3f}%".format(accuracy/len(testloader)*100))

# Save model checkpoint
model.class_to_idx = train_data.class_to_idx
imgCls_checkpoint = {'class_to_idx': model.class_to_idx,
              'classifier_state':model.classifier.state_dict(),
              'state_dict': model.state_dict(),
              'arch': arch,
              'hidden_units':hidden_units,
              'epochs':epochs,
              'criterion':criterion,
              'optimizer_state': optimizer.state_dict() }

torch.save(imgCls_checkpoint, filename)

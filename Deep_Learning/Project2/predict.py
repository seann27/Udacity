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
import json
from CustomClassifiers import Classifier

# Define mapping for flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(file):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(file, map_location=map_location)
    model = pretrained_models.getPretrainedModel(checkpoint['arch'])
    hidden_units = checkpoint['hidden_units']
    if hidden_units:
        model.classifier = Classifier(hidden_units)
    model.classifier.load_state_dict(checkpoint['classifier_state'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.resize((256,256))
    top = (256-224)/2
    left = (256-224)/2
    bottom = (256+224)/2
    right = (256+224)/2
    image = image.crop((left,top,right,bottom))
    np_image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose()
    return np_image

parser = argparse.ArgumentParser(description="Load model and hyperparameters")
parser.add_argument('--image', help='Filepath of image')
parser.add_argument('--topk', help='Number of top predictions to display')
args = parser.parse_args()

image_path = args.image
if image_path is None:
    print("Error! No image path specified!")
    exit()

# set default
topk = 5
if args.topk:
    topk = int(args.topk)

filename = 'vgg19_checkpoint.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device: ",device)
model = load_checkpoint(filename)
model.to(device)

def predict(image_path, model, topk=5):
    image = Image.open(image_path)
    np_image = process_image(image)
    np_tensor = torch.from_numpy(np_image)
    np_tensor.unsqueeze_(0)
    np_tensor = np_tensor.float()
    np_tensor = np_tensor.to(device)
    log_ps = model.forward(np_tensor)
    ps = torch.exp(log_ps)
    top_p,top_class = ps.topk(topk,dim=1)
    return top_p,top_class

with torch.no_grad():
    top_p,top_class = predict(image_path,model,topk)
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    top_p = top_p.numpy().reshape(-1)
    top_class = top_class.numpy().reshape(-1)
    print(top_p)
    print(top_class)

    # invert the flower name mappings
    inverted_dict = dict([[v,k] for k,v in model.class_to_idx.items()])

    for p, name in zip(top_p,top_class):
        idx = inverted_dict[int(name)]
        print(cat_to_name[str(idx)],': ',p)

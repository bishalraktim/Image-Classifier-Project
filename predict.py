# import python packages
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim

import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np

import PIL
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import copy
import os
import json

import argparse

parser = argparse.ArgumentParser(description = "PredictImages")

# adding all the command line arguments
parser.add_argument('--input_image', action = 'store', 
                    default = '/home/workspace/ImageClassifier/flowers/test/11/image_03177.jpg', type = str)
parser.add_argument('--checkpoint', action = 'store', default = 'checkpoint.pth', type = str)
parser.add_argument('--top_k', action = 'store', default = 5, dest = 'top_k', type = int)
parser.add_argument('--gpu', action = 'store', default = False, dest = 'gpu', type = bool)


# printing and accessing values of all the aguments
args = parser.parse_args()
    
print("Argument 1:", args.input_image)
print("Argument 2:", args.checkpoint)
print("Argument 3:", args.top_k)
print("Argument 4:", args.gpu)

print(" ")

image_path = args.input_image
path = args.checkpoint
total_class_no = args.top_k
power = args.gpu


# defining and printing the device
if power == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} - GPU is available!")
else:
    device = torch.device('cpu')
    print(f"{device} - CPU is available!")


# defining the function for loading the data
root_path = '/home/workspace/ImageClassifier/' 
data_dir = root_path + 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
# Defining the transforms for the training, validation, and testing sets
training_transform = transforms.Compose([transforms.RandomRotation(35), transforms.RandomResizedCrop(224), 
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])]) 
    
validation_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                                           transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                                                       [0.229, 0.224, 0.225])]) 
    
testing_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
data_transforms = [training_transform, validation_transform, testing_transform] 
    
    
# Loading the datasets with ImageFolder 
training_dataset = datasets.ImageFolder(train_dir, transform=data_transforms[0]) 
validation_dataset = datasets.ImageFolder(train_dir, transform=data_transforms[1])
testing_dataset = datasets.ImageFolder(train_dir, transform=data_transforms[2])
    
image_datasets = [training_dataset, validation_dataset, testing_dataset]
    
    
# Defining the dataloaders
training_dataloader = torch.utils.data.DataLoader(image_datasets[0], batch_size=32, shuffle=True) 
validation_dataloader = torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True) 
testing_dataloader = torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True) 
    
dataloaders = [training_dataloader, validation_dataloader, testing_dataloader]


# Label mapping
root_path = '/home/workspace/ImageClassifier/'
with open(root_path + 'cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


archs = {"vgg16" : 25088, "densenet121" : 1024, "alexnet" : 9216}

def new_network(arch, dropout, hl, lnr):
    if arch == 'vgg16': 
        model = models.vgg16(pretrained=True) 
        
    elif arch == 'densenet121': 
        model = models.densenet121(pretrained=True)
        
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        
    else:
        print(f"Hey, {arch} is an invalid model. Could you please enter a valid model?")
            
            
    # Freezing parameters
    for param in model.parameters(): 
        param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('dropout', nn.Dropout(dropout)), 
                                                ('input', nn.Linear(archs.get(arch), hl)),
                                                ('relu1', nn.ReLU()), 
                                                ('hl1', nn.Linear(hl, 110)), 
                                                ('relu2', nn.ReLU()), 
                                                ('hl2', nn.Linear(110, 80)), 
                                                ('relu3',nn.ReLU()), 
                                                ('hl3', nn.Linear(80, 104)), 
                                                ('output', nn.LogSoftmax(dim=1))])) 
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lnr)
        model.to(device)
        
        return model, optimizer, criterion


# defining the function for loading the checkpoint
def loading_checkpoint(file_path):
    checkpoint = torch.load(file_path) 
    arch = checkpoint.get('arch') 
    dropout = checkpoint.get('dropout')
    hl = checkpoint.get('hl')
    lnr = checkpoint.get('lnr')
    
    model,_,_ = new_network(arch, dropout, hl, lnr) 
    model.class_to_idx = checkpoint.get('class_to_idx') 
    model.load_state_dict(checkpoint.get('state_dict'))
    return model

model = loading_checkpoint(path)  
print(model)
  

# defining the process image function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    adjust = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = adjust(image)
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# defining the function for class prediction
def predict(image_dir, model, total_class_no):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = Image.open(image_dir)
    image = process_image(image)
    
    # Converting 2D image vector to 1D
    image = np.expand_dims(image, 0)
    
    
    image = torch.from_numpy(image)
    
    model.eval()
    inps = Variable(image).to(device)
    logs = model.forward(inps)
    
    probability = F.softmax(logs, dim=1)
    topk = probability.cpu().topk(total_class_no)
    
    return (k.data.numpy().squeeze().tolist() for k in topk)


# define and print the class
def printing_results(): 
    total_class = image_datasets[0].classes

    print(f"\nDisplaying the most likely image classes:\n{total_class}")

    probability, classes = predict(image_path, model.to(device), total_class_no)
    
    print(f"\nDisplaying probabilities:\n{probability}")
    print(f"\nDisplaying topk classes:\n{classes}")
    flowers = [cat_to_name[total_class[k]] for k in classes]
    print(f"\nMapping the class values to category names:\n{flowers}")
    print(" ")
   
    print("All done!")
    
printing_results()
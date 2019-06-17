# import the packages
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

from collections import OrderedDict
import time
import copy
import os

import argparse

parser = argparse.ArgumentParser(description = "TrainingModel")

# adding all the command line arguments
parser.add_argument('--arch', action = 'store', default = 'alexnet', dest = 'arch', type = str)
parser.add_argument('--dropout', action = "store", default = 0.6, dest = 'dropout', type = float)
parser.add_argument('--hidden_units', action = 'store', default = 140, dest = 'hidden_units', type = int)
parser.add_argument('--learning_rate', action = 'store', default = 0.001, dest = 'learning_rate', type = float)
parser.add_argument('--gpu', action = 'store', default = False, dest = 'gpu', type = bool)
parser.add_argument('--epochs', action = 'store', default = 1, dest = 'epochs', type = int)
parser.add_argument('--checkpoint', action = 'store', default = 'checkpoint.pth', dest = 'checkpoint', type = str)
parser.add_argument('--print_every', action = 'store', default = '50', dest = 'print_every', type = int)


# printing and accessing values of all the aguments
args = parser.parse_args()
    
print("Argument 1:", args.arch)
print("Argument 2:", args.dropout)
print("Argument 3:", args.hidden_units)
print("Argument 4:", args.learning_rate)
print("Argument 5:", args.gpu)
print("Argument 6:", args.epochs)
print("Argument 7:", args.checkpoint)
print("Argument 8:", args.print_every)

print(" ")

arch = args.arch
dropout = args.dropout
hl = args.hidden_units
lnr = args.learning_rate
power = args.gpu
epochs = args.epochs
file_path = args.checkpoint
print_every = args.print_every


# defining and printing the device
if power == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} - GPU is available for training the model!")
else:
    device = torch.device('cpu')
    print(f"{device} - CPU is available for training the model!")    
    
    
archs = {"vgg16" : 25088, "densenet121" : 1024, "alexnet" : 9216}

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
model, optimizer, criterion = new_network(arch, dropout, hl, lnr)


# defining the training network
starts = time.time()
def training_the_network(model, optimizer, criterion, epochs, print_every):
    steps = 0
    running_loss = 0
    
    model.to(device)
    
    print('Training Starts:')
    
    for ep in range(epochs):
        for inp, (inputs, labels) in enumerate(dataloaders[0]): 
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # forward pass and backward pass
            outs = model.forward(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item() 
            
            if steps % print_every == 0: 
                model.eval()
                testing_loss = 0
                accuracy = 0 
                
                for inp, (inputs1,labels1) in enumerate(dataloaders[1]): 
                    optimizer.zero_grad()

                    inputs1, labels1 = inputs1.to(device) , labels1.to(device)
                    model.to(device)
                    
                    with torch.no_grad(): 
                        outs = model.forward(inputs1)
                        testing_loss = criterion(outs,labels1)
                        ps = torch.exp(outs).data
                        equality = (labels1.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean() 
                    
                testing_loss = testing_loss / len(dataloaders[1]) 
                accuracy = accuracy /len(dataloaders[1]) 
                    
                print("Epochs: {}/{}... ".format(ep+1, epochs), 
                      "Training Loss: {:.5f}".format(running_loss/print_every), 
                      "Validation Loss {:.5f}".format(testing_loss), 
                      "Accuracy: {:.5f}".format(accuracy)) 
                    
                running_loss = 0
                
training_the_network(model, optimizer, criterion, epochs, print_every)
time_taken = time.time() - starts
print("\nTotal time for training: {:.0f}m {:.0f}s".format(time_taken//60, time_taken % 60))


# defining the accuracy testing
def accuracy_testing(check):
    correct = 0
    total = 0
    model.to(device)
    
    with torch.no_grad():
        for datas in dataloaders[2]:
            imgs, lbls = datas
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs) 
            _, predicted = torch.max(outs.data, 1) 
            total += lbls.size(0) 
            correct += (predicted == lbls).sum().item() 
    print(f"Network's accuracy on the test images: {100 * correct / total}%")

print("Accuracy test is starting .......")    
accuracy_testing(dataloaders[2])


# defining the fuction for saving checkpoint
def saving_the_checkpoint(file_path, arch, hl, dropout, lnr, epochs):
    model.class_to_idx = image_datasets[0].class_to_idx 
    
    checkpoint = {'file_path' : file_path, 'arch' : arch, 'hl' : hl, 'dropout' : dropout, 
                  'lnr' : lnr, 'epochs' : epochs, 
                  'state_dict': model.state_dict(), 
                  'class_to_idx': model.class_to_idx} 
    torch.save(checkpoint, file_path)
    
saving_the_checkpoint(file_path, arch, hl, dropout, lnr, epochs)
print(f"Great! The checkpoint is saved as {file_path} for making predictions.") 
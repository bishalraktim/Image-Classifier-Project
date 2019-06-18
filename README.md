# Image-Classifier-Project

Developing an AI application
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

In [1]:
# Imports here
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from PIL import Image
from collections import OrderedDict

import time

import json
import copy
import PIL
import os
Load the data
Here you'll use torchvision to load the data (documentation). The data should be included alongside this notebook, otherwise you can download it here. The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.

In [2]:
# defining, checking and printing the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is available for training the model!")
cuda:0 is available for training the model!
In [3]:
pwd
Out[3]:
'/home/workspace/aipnd-project'
In [4]:
root_path = '/home/workspace/aipnd-project/'
data_dir = root_path + 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
In [5]:
# TODO: Define your transforms for the training, validation, and testing sets
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


# TODO: Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder(train_dir, transform=data_transforms[0])
validation_dataset = datasets.ImageFolder(train_dir, transform=data_transforms[1])
testing_dataset = datasets.ImageFolder(train_dir, transform=data_transforms[2])

image_datasets = [training_dataset, validation_dataset, testing_dataset]


# TODO: Using the image datasets and the trainforms, define the dataloaders
training_dataloader = torch.utils.data.DataLoader(image_datasets[0], batch_size=32, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True)
testing_dataloader = torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True)

dataloaders = [training_dataloader, validation_dataloader, testing_dataloader]


print(f"> {data_transforms} \n")
print(f">> {image_datasets} \n")
print(f">>> {dataloaders} \n")
> [Compose(
    RandomRotation(degrees=(-35, 35), resample=False, expand=False)
    RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BILINEAR)
    RandomHorizontalFlip(p=0.5)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
), Compose(
    Resize(size=256, interpolation=PIL.Image.BILINEAR)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
), Compose(
    Resize(size=256, interpolation=PIL.Image.BILINEAR)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)] 

>> [Dataset ImageFolder
    Number of datapoints: 6552
    Root Location: /home/workspace/aipnd-project/flowers/train
    Transforms (if any): Compose(
                             RandomRotation(degrees=(-35, 35), resample=False, expand=False)
                             RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BILINEAR)
                             RandomHorizontalFlip(p=0.5)
                             ToTensor()
                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         )
    Target Transforms (if any): None, Dataset ImageFolder
    Number of datapoints: 6552
    Root Location: /home/workspace/aipnd-project/flowers/train
    Transforms (if any): Compose(
                             Resize(size=256, interpolation=PIL.Image.BILINEAR)
                             CenterCrop(size=(224, 224))
                             ToTensor()
                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         )
    Target Transforms (if any): None, Dataset ImageFolder
    Number of datapoints: 6552
    Root Location: /home/workspace/aipnd-project/flowers/train
    Transforms (if any): Compose(
                             Resize(size=256, interpolation=PIL.Image.BILINEAR)
                             CenterCrop(size=(224, 224))
                             ToTensor()
                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         )
    Target Transforms (if any): None] 

>>> [<torch.utils.data.dataloader.DataLoader object at 0x7f537bbfacc0>, <torch.utils.data.dataloader.DataLoader object at 0x7f537bbfaef0>, <torch.utils.data.dataloader.DataLoader object at 0x7f537bb158d0>] 

Label mapping
You'll also need to load in a mapping from category label to category name. You can find this in the file cat_to_name.json. It's a JSON object which you can read in with the json module. This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

In [6]:
# Label mapping

# import json

with open(root_path + 'cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    print(cat_to_name)
{'21': 'fire lily', '3': 'canterbury bells', '45': 'bolero deep blue', '1': 'pink primrose', '34': 'mexican aster', '27': 'prince of wales feathers', '7': 'moon orchid', '16': 'globe-flower', '25': 'grape hyacinth', '26': 'corn poppy', '79': 'toad lily', '39': 'siam tulip', '24': 'red ginger', '67': 'spring crocus', '35': 'alpine sea holly', '32': 'garden phlox', '10': 'globe thistle', '6': 'tiger lily', '93': 'ball moss', '33': 'love in the mist', '9': 'monkshood', '102': 'blackberry lily', '14': 'spear thistle', '19': 'balloon flower', '100': 'blanket flower', '13': 'king protea', '49': 'oxeye daisy', '15': 'yellow iris', '61': 'cautleya spicata', '31': 'carnation', '64': 'silverbush', '68': 'bearded iris', '63': 'black-eyed susan', '69': 'windflower', '62': 'japanese anemone', '20': 'giant white arum lily', '38': 'great masterwort', '4': 'sweet pea', '86': 'tree mallow', '101': 'trumpet creeper', '42': 'daffodil', '22': 'pincushion flower', '2': 'hard-leaved pocket orchid', '54': 'sunflower', '66': 'osteospermum', '70': 'tree poppy', '85': 'desert-rose', '99': 'bromelia', '87': 'magnolia', '5': 'english marigold', '92': 'bee balm', '28': 'stemless gentian', '97': 'mallow', '57': 'gaura', '40': 'lenten rose', '47': 'marigold', '59': 'orange dahlia', '48': 'buttercup', '55': 'pelargonium', '36': 'ruby-lipped cattleya', '91': 'hippeastrum', '29': 'artichoke', '71': 'gazania', '90': 'canna lily', '18': 'peruvian lily', '98': 'mexican petunia', '8': 'bird of paradise', '30': 'sweet william', '17': 'purple coneflower', '52': 'wild pansy', '84': 'columbine', '12': "colt's foot", '11': 'snapdragon', '96': 'camellia', '23': 'fritillary', '50': 'common dandelion', '44': 'poinsettia', '53': 'primula', '72': 'azalea', '65': 'californian poppy', '80': 'anthurium', '76': 'morning glory', '37': 'cape flower', '56': 'bishop of llandaff', '60': 'pink-yellow dahlia', '82': 'clematis', '58': 'geranium', '75': 'thorn apple', '41': 'barbeton daisy', '95': 'bougainvillea', '43': 'sword lily', '83': 'hibiscus', '78': 'lotus lotus', '88': 'cyclamen', '94': 'foxglove', '81': 'frangipani', '74': 'rose', '89': 'watercress', '73': 'water lily', '46': 'wallflower', '77': 'passion flower', '51': 'petunia'}
Building and training the classifier
Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from torchvision.models to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to the rubric for guidance on successfully completing this section. Things you'll need to do:

Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
Train the classifier layers using backpropagation using the pre-trained network to get the features
Track the loss and accuracy on the validation set to determine the best hyperparameters
We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

Note for Workspace users: If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with ls -lh), you should reduce the size of your hidden layers and train again.

In [7]:
# TODO: Build and train your network
archs = {"vgg16" : 25088, "densenet121" : 1024, "alexnet" : 9216}


def new_network(arch = 'vgg16', dropout = 0.6, hl = 140, lnr = 0.001):
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
                                                ('output', nn.LogSoftmax(dim=1)),
                                               ])) 
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lnr)
        model.to(device)
        
        return model, optimizer, criterion 
model, optimizer, criterion = new_network('alexnet')
model
Downloading: "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth" to /root/.torch/models/alexnet-owt-4df8aa71.pth
100%|██████████| 244418560/244418560 [00:17<00:00, 14103253.02it/s]
Out[7]:
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (dropout): Dropout(p=0.6)
    (input): Linear(in_features=9216, out_features=140, bias=True)
    (relu1): ReLU()
    (hl1): Linear(in_features=140, out_features=110, bias=True)
    (relu2): ReLU()
    (hl2): Linear(in_features=110, out_features=80, bias=True)
    (relu3): ReLU()
    (hl3): Linear(in_features=80, out_features=104, bias=True)
    (output): LogSoftmax()
  )
)
In [8]:
epochs = 5
steps = 0
running_loss = 0
print_every = 50

model.to(device)

starts = time.time()
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
            
time_taken = time.time() - starts
print("\nTotal time for training: {:.0f}m {:.0f}s".format(time_taken//60, time_taken % 60))
Training Starts:
Epochs: 1/5...  Training Loss: 4.28353 Validation Loss 0.01709 Accuracy: 0.21453
Epochs: 1/5...  Training Loss: 3.32951 Validation Loss 0.00975 Accuracy: 0.35610
Epochs: 1/5...  Training Loss: 2.68243 Validation Loss 0.01097 Accuracy: 0.51275
Epochs: 1/5...  Training Loss: 2.17130 Validation Loss 0.01154 Accuracy: 0.56128
Epochs: 2/5...  Training Loss: 1.89089 Validation Loss 0.00637 Accuracy: 0.65290
Epochs: 2/5...  Training Loss: 1.75805 Validation Loss 0.00425 Accuracy: 0.69263
Epochs: 2/5...  Training Loss: 1.59525 Validation Loss 0.00507 Accuracy: 0.70483
Epochs: 2/5...  Training Loss: 1.51218 Validation Loss 0.00499 Accuracy: 0.75279
Epochs: 3/5...  Training Loss: 1.44301 Validation Loss 0.00484 Accuracy: 0.73115
Epochs: 3/5...  Training Loss: 1.43553 Validation Loss 0.00396 Accuracy: 0.75030
Epochs: 3/5...  Training Loss: 1.35622 Validation Loss 0.00386 Accuracy: 0.75589
Epochs: 3/5...  Training Loss: 1.38151 Validation Loss 0.00263 Accuracy: 0.78679
Epochs: 4/5...  Training Loss: 1.24596 Validation Loss 0.00234 Accuracy: 0.76916
Epochs: 4/5...  Training Loss: 1.13941 Validation Loss 0.00266 Accuracy: 0.81489
Epochs: 4/5...  Training Loss: 1.23211 Validation Loss 0.00451 Accuracy: 0.82444
Epochs: 4/5...  Training Loss: 1.19394 Validation Loss 0.00191 Accuracy: 0.81601
Epochs: 5/5...  Training Loss: 1.11999 Validation Loss 0.00315 Accuracy: 0.83176
Epochs: 5/5...  Training Loss: 1.02603 Validation Loss 0.00295 Accuracy: 0.84304
Epochs: 5/5...  Training Loss: 1.10328 Validation Loss 0.00443 Accuracy: 0.83277
Epochs: 5/5...  Training Loss: 1.16171 Validation Loss 0.00129 Accuracy: 0.82327

Total time for training: 28m 54s
Testing your network
It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

In [9]:
# TODO: Do validation on the test set
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
    
accuracy_testing(dataloaders[2])
Network's accuracy on the test images: 84.92063492063492%
Save the checkpoint
Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: image_datasets['train'].class_to_idx. You can attach this to the model as an attribute which makes inference easier later on.

model.class_to_idx = image_datasets['train'].class_to_idx

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, optimizer.state_dict. You'll likely want to use this trained model in the next part of the project, so best to save it now.

In [11]:
# TODO: Save the checkpoint 
model.class_to_idx = image_datasets[0].class_to_idx

checkpoint = {'arch': 'alexnet', 'dropout' : 0.6, 'hl' : 140, 'lnr' : 0.001, 
              'epochs' : 6, 'state_dict': model.state_dict(), 
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, 'checkpoint.pth')
Loading the checkpoint
At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

In [12]:
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(file_path):
    checkpoint = torch.load('checkpoint.pth') 
    arch = checkpoint.get('arch')
    dropout = checkpoint.get('dropout')
    hl = checkpoint.get('hl')
    lnr = checkpoint.get('lnr') 
    
    model,_,_ = new_network(arch, dropout, hl, lnr) 
    model.class_to_idx = checkpoint.get('class_to_idx') 
    model.load_state_dict(checkpoint.get('state_dict'))
    
    return model

model = loading_checkpoint('checkpoint.pth')  
print(model)
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (dropout): Dropout(p=0.6)
    (input): Linear(in_features=9216, out_features=140, bias=True)
    (relu1): ReLU()
    (hl1): Linear(in_features=140, out_features=110, bias=True)
    (relu2): ReLU()
    (hl2): Linear(in_features=110, out_features=80, bias=True)
    (relu3): ReLU()
    (hl3): Linear(in_features=80, out_features=104, bias=True)
    (output): LogSoftmax()
  )
)
Inference for classification
Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called predict that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like

probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
First you'll need to handle processing the input image such that it can be used in your network.

Image Preprocessing
You'll want to use PIL to load the image (documentation). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training.

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the thumbnail or resize methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so np_image = np.array(pil_image).

As before, the network expects the images to be normalized in a specific way. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. You'll want to subtract the means from each color channel, then divide by the standard deviation.

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using ndarray.transpose. The color channel needs to be first and retain the order of the other two dimensions.

In [13]:
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model 
    adjust = transforms.Compose([transforms.Resize(256), 
                                 transforms.CenterCrop(224), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                      std=[0.229, 0.224, 0.225])]) 
    
    image = adjust(image) 
    return image
To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your process_image function works, running the output through this function should return the original image (except for the cropped out portions).

In [14]:
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
In [15]:
with Image.open(root_path + 'flowers/valid/11/image_03100.jpg') as img:
    plt.imshow(img)
model.class_to_idx = image_datasets[0].class_to_idx

Class Prediction
Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use x.topk(k). This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using class_to_idx which hopefully you added to the model or from an ImageFolder you used to load the data (see here). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
In [16]:
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file 
    image = Image.open(image_path) 
    image = process_image(image)
    
    # Converting 2D image vector to 1D
    image = np.expand_dims(image, 0)
    
    image = torch.from_numpy(image)
    
    model.eval()
    inps = Variable(image).to(device)
    logs = model.forward(inps)
    
    probability = F.softmax(logs, dim=1)
    topk = probability.cpu().topk(topk)
    
    return (k.data.numpy().squeeze().tolist() for k in topk)
In [17]:
# print the class

total_class = image_datasets[0].classes

print(total_class)
['1', '10', '100', '101', '102', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
In [19]:
image_dir = root_path + 'flowers/test/100/image_07902.jpg'
probability, classes = predict(image_dir, model.to(device))
print(probability, "\n")
print(classes, "\n")
flowers = [cat_to_name[total_class[k]] for k in classes]
print(flowers)
[0.7814573049545288, 0.09489908069372177, 0.08821138739585876, 0.016098378226161003, 0.0060163214802742004] 

[2, 77, 11, 38, 34] 

['blanket flower', 'passion flower', 'purple coneflower', 'barbeton daisy', 'great masterwort']
Sanity Checking
Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use matplotlib to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the cat_to_name.json file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the imshow function defined above.

In [20]:
# TODO: Display an image along with the top 5 classes
def check_sanity(image_dir, probability, classes, maps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    image = Image.open(image_dir)
    
    fig, (axn, axm) = plt.subplots(figsize=(8, 11), ncols=1, nrows=2)
    flower = maps[image_dir.split('/')[-2]]
    axn.set_title(flower)
    axn.imshow(image)
    axn.axis('off')
    
    y_dir = np.arange(len(probability))
    axm.barh(y_dir, probability, align='center')
    axm.set_yticks(y_dir)
    axm.set_yticklabels(flowers)
    axm.invert_yaxis() 
    
check_sanity(image_dir, probability, classes, cat_to_name)

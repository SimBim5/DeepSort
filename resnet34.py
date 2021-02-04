from torchvision import models
import torch
import torch.nn as nn


    ## All pre-trained models expect input images normalized in the same way
    ## normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])



    cnn = torchvision.models.resnet34(pretrained=True, progress=True, **kwargs)
    
    ##freeze last layer
    cnn = torch.nn.Sequential(*(list(cnn.children())[:-1]))

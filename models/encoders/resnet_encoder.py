import torch.nn as nn
import torch
import torchvision
from .trajectory_encoder import EncoderSelfAttention

def Resnet(pretrain=True, layers_to_unfreeze=8, layers_to_delete=2, in_planes=3):
    """
    param:
        pretrain: Define if we load a pretrained model from ImageNet
        layers_to_unfreeze: Define the number of layers that we want to train at the end of the Resnet
        layers_to_delete: Define the numbers of layers that we want to delete
        in_planes: Define the numbers of input channels of images (supported values: 1,2 or 3)
    return: The Resnet model
    """
    resnet = torchvision.models.resnet18(pretrained=pretrain)
    # Create a new model cause we don't want the pooling operation at the end and the classifier
    model = nn.Sequential()
    number_of_layers = len(list(resnet.children())) - layers_to_delete # In practice it remove the pooling operation and the classifier

    if number_of_layers<layers_to_unfreeze:
        layers_to_unfreeze = number_of_layers
    layers_to_freeze = number_of_layers - layers_to_unfreeze
    i=0
    for child in resnet.children():
        # For the first layers we create a new weight if in_planes is not 3 cause ResNet is pretrain on image with 3 channels there is no version for 1 channel
        if i==0 and in_planes<3:
            if i<layers_to_freeze: # Define if we freeze this layer or no
                for param in child.parameters():
                    param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
            w = child._parameters['weight'].data # Get the weight for 3 channels data
            child._modules['0'] = nn.Conv2d(in_planes, 64, kernel_size=3, padding=1) # Define the new conv layer
            if in_planes == 1:
                child._parameters['weight'].data = w.mean(dim=1, keepdim=True) # If the number of channels is 1 we made the mean of channels to set the new weight
            else:
                child._parameters['weight'].data = w[:, :-1] * 1.5

        if i<layers_to_freeze: # Define if we freeze this layer or no
            for param in child.parameters():
                param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
        if i<number_of_layers: # To define if we keep this layer or not
            model.append(child) 
        i+=1
    return model

class features_extraction(nn.Module):
    """
    param:
    conv_model: The convolution model used before capsules for the moment only ResNet is supported
    in_planes: Numbers of channels for the image
    """
    def __init__(self,conv_model,in_planes: int):
        super().__init__()
        self.conv_model = conv_model
        self.in_planes = in_planes
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,input):
        shape = input.size()
        x = input.view(-1,self.in_planes,shape[-2],shape[-1])
        x = self.conv_model(x)
        x = self.pooling(x)
        return x
    
class _GestureTransformer(nn.Module):
    """Multi-Modal model on 3 or 1 channel"""
    def __init__(self,device,backbone="resnet",in_planes=3,pretrained= True,input_dim=512,layers_to_unfreeze=8,layers_to_delete=2,n_head=8,n_module=6,ff_size=1024,dropout1d=0.5):
        super(_GestureTransformer, self).__init__()

        self.in_planes = in_planes
        self.device = device
        self.conv_name = backbone
        self.conv_model = None
        
        if self.conv_name.lower()=="resnet":
            self.conv_model = Resnet(pretrained,layers_to_unfreeze=2,layers_to_delete=layers_to_delete,in_planes=in_planes)
        else:
            raise NotImplementedError("The model {} is not supported!".format(self.conv_name))
        self.conv_model.to(device)
        self.features = features_extraction(self.conv_model,in_planes)

        self.self_attention = EncoderSelfAttention(device,input_dim,64,64,n_head=n_head,dff=ff_size,dropout_transformer=dropout1d,n_module=n_module)

        self.pool = nn.AdaptiveAvgPool2d((8,input_dim)) #final pooling

    def forward(self, x):
        shape = x.shape
        x = self.features(x)
        x = x.view(shape[0],shape[1],-1)
        x = self.self_attention(x)

        x = self.pool(x).squeeze(dim=1) #final pooling
        return x

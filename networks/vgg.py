from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=False).features
        checkpoint = torch.load('./pretrained_models/vgg16.pth')
        self.loadVggWights(checkpoint)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def loadVggWights(self, checkpoint):
        tmp_chek_layer_list = list(checkpoint.keys())
        modelState = self.vgg_pretrained_features.state_dict()
        newdict = {}
        for index, item in enumerate(modelState.keys()):
            newdict[item] = checkpoint[tmp_chek_layer_list[index]]
        self.vgg_pretrained_features.load_state_dict(newdict)


    def forward(self, X, layer):

        output = []

        h = self.slice1(X)
        h_relu1_2 = h
        output.append(h_relu1_2)
        if layer == 0:
            return output

        h = self.slice2(h)
        h_relu2_2 = h
        output.append(h_relu2_2)
        if layer == 1:
            return output

        h = self.slice3(h)
        h_relu3_3 = h
        output.append(h_relu3_3)
        if layer == 2:
            return output

        h = self.slice4(h)
        h_relu4_3 = h
        output.append(h_relu4_3)
        if layer == 3:
            return output

        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        #return out

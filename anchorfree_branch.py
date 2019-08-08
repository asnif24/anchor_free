import numpy as np
import torch
import torch.nn as nn


# class AnchorFree(nn.Module):
#     """docstring for AnchorFree"""
#     def __init__(self, arg):
#         super(AnchorFree, self).__init__()
#         self.arg = arg
        

#     def forward(self):



# class RegressionModel(nn.Module):
#     def __init__(self, num_features_in, feature_size=256):
#         super(RegressionModel, self).__init__()

#         self.output = nn.Conv2d(num_features_in, 4, kernel_size=3, padding=1)
#         self.act = nn.ReLU()

#     def forward(self, x):


#         out = self.output(x)
#         out = self.act(out)

#         # out is B x C x W x H, with C = 4
#         out = out.permute(0, 2, 3, 1)

#         return out.contiguous().view(out.shape[0], -1, 4)

# class ClassificationModel(nn.Module):
#     def __init__(self, num_features_in, num_classes=80, prior=0.01, feature_size=256):
#         super(ClassificationModel, self).__init__()

#         self.num_classes = num_classes

#         self.output = nn.Conv2d(num_features_in, num_classes, kernel_size=3, padding=1)
#         self.output_act = nn.Sigmoid()

#     def forward(self, x):

#         out = self.output(x)
#         out = self.output_act(out)

#         # out is B x C x W x H, with C = n_classes
#         out1 = out.permute(0, 2, 3, 1)

#         batch_size, width, height, channels = out1.shape

#         # out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
#         out2 = out1.view(batch_size, width, height, self.num_classes)

#         return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, 4, kernel_size=3, padding=1)
        self.output_act = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = 4
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        # self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        # out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out2 = out1.view(batch_size, width, height, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)




        
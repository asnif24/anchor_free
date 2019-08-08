import numpy as np
import torch
import torch.nn as nn
from anchorfree_losses import FocalLoss, IoULoss


# regression = [self.regressionModel(feature) for feature in features]
# classification = [self.classificationModel(feature) for feature in features]

class FeatureSelection(nn.Module):
    def __init__(self):
        self.focal_loss = FocalLoss()
        self.IoU_loss = IoULoss()


    def forward(self, regression, classification):
        # len(classification) = 5
        total_loss = []
        for layer in range(len(classification)):
            cls_loss = self.focal_loss(classification[layer])
            reg_loss = self.IoU_loss(regression[layer])
            total_loss.append(cls_loss+reg_loss)

        total_loss = torch.tensor(total_loss)
        min_layer = torch.argmin(total_loss, dim=0)

        return min_layer













































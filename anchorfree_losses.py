import numpy as np
import torch
import torch.nn as nn

from math import floor

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


# def box_projection(box, layer):
#     # box: [x, y, w, h]
#     # box_projections = []
#     w, h = box[2], box[3]
#     for i in range(layer):
#         w = w//2
#         h = h//2
#     return [box[0], box[1], w, h]

# def box_projection(box, layer):
#     # box: [x, y, w, h]
#     # box_projections = []
#     x, y, w, h = box[0], box[1], box[2], box[3]
#     for i in range(layer):
#         x = x//2
#         y = y//2
#         w = w//2
#         h = h//2
#     return [x, y, w, h]


effctive_factor = 0.2
ignoring_factor = 0.5

# CLASSIFICATION

def boxScaling(box, scale_factor):
    x = (box[:,0]+box[:,2])/2
    y = (box[:,1]+box[:,3])/2
    w = box[:,2]-box[:,0]
    h = box[:,3]-box[:,1]
    return torch.tensor([x-w*scale_factor/2, y-h*scale_factor/2, x+w*scale_factor/2, y+h*scale_factor/2])
 

def focalloss(classification, bbox):
    alpha = 0.25
    gamma = 2.0
    
    effective = 0.2
    ignoring = 0.5
    


    
    effective_box = boxScaling(bbox[0:4], effective)
    ignoring_box = boxScaling(bbox[0:4], ignoring)

    targets = torch.ones(classification.shape)*(-1)
    class_index = bbox[4].int()

    targets[floor(bbox[0]):floor(bbox[2])+1, floor(bbox[1]):floor(bbox[3])+1, :] = 0
    targets[floor(ignoring_box[0]):floor(ignoring_box[2])+1, floor(ignoring_box[1]):floor(ignoring_box[3])+1, :] = -1
    targets[floor(effective_box[0]):floor(effective_box[2])+1, floor(effective_box[1]):floor(effective_box[3])+1, :] = 0
    targets[floor(effective_box[0]):floor(effective_box[2])+1, floor(effective_box[1]):floor(effective_box[3])+1, class_index] = 1
    
    non_ignore = torch.sum(targets!=-1).float()
    
    alpha_factor = torch.ones(targets.shape)* alpha
    alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
    focal_weight = torch.where(torch.eq(targets, 1), 1. - classification, classification)
    
    bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
    cls_loss = focal_weight * bce
    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

    average_loss = cls_loss.sum()/non_ignore

    return cls_loss, non_ignore, average_loss

class FocalLoss(nn.Module):
    def forward(self, classifications, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]       # batch_size = 1
        classification_losses = []

        for j in range(batch_size):
            classification = classifications[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            # classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()
            

# REGRESSION

def getIoU(box1,box2):
    area1 = (box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])
    area2 = (box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])
    
    inter_w = (box1[:,:,2]-box1[:,:,0])+(box2[:,:,2]-box2[:,:,0])-(torch.max(box1[:,:,2],box2[:,:,2])-torch.min(box1[:,:,0],box2[:,:,0]))
    inter_h = (box1[:,:,3]-box1[:,:,1])+(box2[:,:,3]-box2[:,:,1])-(torch.max(box1[:,:,3],box2[:,:,3])-torch.min(box1[:,:,1],box2[:,:,1]))
    inter_w = torch.clamp(inter_w, min=0)
    inter_h = torch.clamp(inter_h, min=0)
    
    intersection = inter_w * inter_h
    
    ua = area1 + area2 - intersection
    ua = torch.clamp(ua, min=1e-8)
    
    IoU = intersection / ua    
#     return area1, area2,intersection, ua, IoU
    return IoU


class IoULoss(nn.Module):
    
    def offsetToBox(x,y,offsets):
        return torch.FloatTensor([x-offsets[0], y-offsets[1], x+offsets[2], y+offsets[3]])

    def forward(self, regressions, annotations):
        # regressions : WxHx4

    def getRegressionBox(regression):
        regression_box = torch.zeros(regression.shape)
        for x in range(regression.shape[0]):
            for y in range(regression.shape[1]):
                regression_box[x][y] = offsetToBox(x,y,regression[x, y, :])
        return regression_box

    def IoUloss(regression, bbox):
        S = 4.0 #normalization constant
        effective = 0.2
        
        effective_box = boxScaling(bbox[:,0:4], effective)
        
        offset_box = boxScaling(bbox[:,0:4], 1/S)
        offset_box = offset_box.unsqueeze(dim=0).unsqueeze(dim=0)
        
        targets = torch.ones(regression.shape[0:2])*(-1)
        targets[floor(effective_box[0]):floor(effective_box[2])+1, floor(effective_box[1]):floor(effective_box[3])+1] = 1
        
        regression_box = getRegressionBox(regression) 
        regression_box = regression_box
        
        effective_box = effective_box.unsqueeze(dim=0).unsqueeze(dim=0)

        IoU_map = getIoU(regression_box, offset_box)

        IoU_loss = torch.where(torch.eq(targets, 1), IoU_map.log()*(-1), torch.zeros(targets.shape))
        
        num_effective = torch.sum(targets==1).float()
        average_loss = IoU_loss.sum()/num_effective
        
        return average_loss

class TotalLoss(nn.Module):
    def forward(self, classifications, regressions, annotations):
        












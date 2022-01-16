from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from losses import kl_loss
from torch.autograd import Variable
import params as p
from utils import get_cfs_matrix, calculate_mean_iou
import copy
from focal_loss import FocalLoss

def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1>0] = 1
    sum2 = img + mask
    sum2[sum2<2] = 0
    sum2[sum2>=2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0*np.sum(sum2)/np.sum(sum1)

def test(dataloader,net,device = "cpu",num_classes = 2,useMaskBoundary = False):
    cfs_matrix = np.zeros((num_classes, num_classes))
    loss_Focalloss = FocalLoss(gamma=2)
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)
    losses = []
    net.eval()
    loader = tqdm(dataloader)
    with torch.no_grad():
        for idx,(img,mask,mask_boundary) in enumerate(loader):
            # Send to device
            img_deformation = copy.deepcopy(img).to(device)
            img_texture = copy.deepcopy(img).to(device)
            mask = mask.to(device)
            mask_boundary = mask_boundary.to(device)

            # load input to model
            output_mask_texture, output_edge_texture = net(img_texture)
            output_mask_deformation, output_edge_deformation = net(img_deformation)

            # mask loss and edge loss
            loss_mask_texture = loss_Softmax(output_mask_texture, mask)
            loss_edge_texture = loss_Focalloss(output_edge_texture, mask_boundary) * 0.1
            loss_mask_deformation = loss_Softmax(output_mask_deformation, mask)
            loss_edge_deformation = loss_Focalloss(output_edge_deformation, mask_boundary) * 0.1

            # KL  loss
            loss_kl_mask = kl_loss(output_mask_texture, Variable(output_mask_deformation.data, requires_grad=False),
                                   p.TEMPERATURE) * p.ALPHA
            loss_kl_edge = kl_loss(output_edge_texture, Variable(output_edge_deformation.data, requires_grad=False),
                                   p.TEMPERATURE) * p.ALPHA * 0.1

            # total loss
            loss = loss_mask_texture + loss_edge_texture + loss_mask_deformation + loss_edge_deformation + loss_kl_mask + loss_kl_edge

            losses.append(loss.item())
            cfs_matrix += get_cfs_matrix(output_mask_texture.detach(), mask.detach())

            loader.set_postfix(test_loss_batch = loss.item())

    mean_iou = calculate_mean_iou(cfs_matrix)
    net.train()
    return sum(losses)/len(losses),mean_iou
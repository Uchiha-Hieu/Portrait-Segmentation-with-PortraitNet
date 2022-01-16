import torch
import torch.nn as nn
from tqdm import tqdm
from losses import kl_loss
from torch.autograd import Variable
import params as p
from focal_loss import FocalLoss

def train(dataloader, net, optimizer, device="cpu"):
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)
    loss_Focalloss = FocalLoss(gamma=2)
    losses = []
    loader = tqdm(dataloader)
    for idx, (img_deformation, img_texture, mask, mask_boundary) in enumerate(loader):
        #Send to device
        img_deformation = img_deformation.to(device)
        img_texture = img_texture.to(device)
        mask = mask.to(device)
        mask_boundary = mask_boundary.to(device)

        #load input to model
        output_mask_texture,output_edge_texture = net(img_texture)
        output_mask_deformation,output_edge_deformation = net(img_deformation)

        #mask loss and edge loss
        loss_mask_texture = loss_Softmax(output_mask_texture,mask)
        loss_edge_texture = loss_Focalloss(output_edge_texture,mask_boundary)*0.1
        loss_mask_deformation = loss_Softmax(output_mask_deformation,mask)
        loss_edge_deformation = loss_Focalloss(output_edge_deformation,mask_boundary)*0.1

        #KL  loss
        loss_kl_mask = kl_loss(output_mask_texture, Variable(output_mask_deformation.data, requires_grad=False),
                              p.TEMPERATURE) * p.ALPHA
        loss_kl_edge = kl_loss(output_edge_texture, Variable(output_edge_deformation.data, requires_grad=False),
                              p.TEMPERATURE) * p.ALPHA * 0.1

        #total loss
        loss = loss_mask_texture + loss_edge_texture + loss_mask_deformation + loss_edge_deformation + loss_kl_mask + loss_kl_edge

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loader.set_postfix(loss=loss.item())

        losses.append(loss.item())

    # return sum(losses) / len(losses), sum(loss_mask_deformations) / len(loss_mask_deformations), sum(
    #     loss_mask_textures) / len(loss_mask_textures), sum(loss_kls) / len(loss_kls)
    return sum(losses)/len(losses)
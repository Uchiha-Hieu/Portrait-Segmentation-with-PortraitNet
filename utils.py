import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def load_checkpoint(net,optimizer,model_folder,model_name,device):
    # using latest_model is higher priority than using bestmodel in the previous stages
    if model_name.split('.')[2] != "tar":
        raise ValueError("latest model file should be '.tar' file and it saves the latest checkpoint")
    model_path = os.path.join(model_folder,model_name)
    model_checkpoint = torch.load(model_path,map_location=device)
    net.load_state_dict(model_checkpoint["state_dict"])
    optimizer.load_state_dict(model_checkpoint["optimizer"])
    current_epoch = model_checkpoint["trained_num_epochs"]
    val_losses = model_checkpoint["val_losses"] # length = trained_num_epochs,loss list relative to each epoch
    val_iou = model_checkpoint["val_iou"]
    train_losses = model_checkpoint["train_losses"]
    return net,optimizer,current_epoch,train_losses,val_losses,val_iou

def save_checkpoint(state,model_folder,model_name):
    if model_name.split('.')[2] != "tar":
        raise ValueError("Check point which is saved should '.tar' file")
    torch.save(state,os.path.join(model_folder,model_name))

def get_cfs_matrix(pred,target):
    """
    target : torch tensor with shape : (batchsize,img_height,img_width)
    pred : torch tensor with shape : (batchsize,num_classes,img_height,img_width)
    """
    pred=pred.cpu()
    target=target.cpu()
    #Flatten
    target = target.view(-1)
    pred=pred.max(1)[1].view(-1)
    cfs=confusion_matrix(target,pred)
    return confusion_matrix(target,pred)

def calculate_mean_iou(cfs_matrix):
    """
    cfs_matrix : confusion matrix : np array with shape (num_classes,num_classes)
    iou in semantic segmentation is calculated : iou=true_positive/(true_positive+false_positive+false_negative)
    true_positive = elements in diagonal of cfs matrix
    false_positive in each row = sum of each row - true_positive in each row
    false_negative in each col = sum of each col - true_positive in each col
    """
    true_positive=np.diag(cfs_matrix)
    false_positive=np.sum(cfs_matrix,1)-true_positive
    false_negative=np.sum(cfs_matrix,0)-true_positive
    iou=true_positive/(false_negative+false_positive+true_positive+1e-5)
    return np.mean(iou) #Get average of ious for entire classes

import argparse
import os
import torch
import torch.nn as nn
import params as cf
from utils import save_checkpoint,load_checkpoint
from torch.utils.data import DataLoader
from model.PortraitNet import PortraitNet
from dataset.ds1 import dataset_1
from train import train
from test import test

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--dataset_folder",type=str,default="../Datasets/dataset_1/",help="Folder contains dataset")
    parser.add_argument("-o","--output_folder",type=str,default="./model_folder",help="Folder to save the model")
    parser.add_argument("-m","--model_name",type=str,default="best.pth.tar",help="Model name")
    parser.add_argument("-ts","--training_status",type=int,default=0,help="0: train from beginning , 1: continue training proceses from the latest saved model , 2 : continue training process from the best saved model")
    parser.add_argument("-p","--pretrained_model",type=str,help="if training status =1 or 2 , load the pretrained model name")
    parser.add_argument("-e","--epochs",type=int,default=300,help="Epochs")
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device : {}".format(device))

    #Get dataset
    train_dataset = dataset_1(img_npy_path=os.path.join(args.dataset_folder,"img_uint8.npy"),mask_npy_path=os.path.join(args.dataset_folder,"msk_uint8.npy"),split_ratio=0.8,isTrain=True,transform_train=True)
    train_loader = DataLoader(train_dataset,batch_size=cf.BATCHSIZE,shuffle=True,num_workers=cf.NUM_WORKERS)
    val_dataset = dataset_1(img_npy_path=os.path.join(args.dataset_folder,"img_uint8.npy"),mask_npy_path=os.path.join(args.dataset_folder,"msk_uint8.npy"),split_ratio=0.8,isTrain=False,transform_train=False)
    val_loader = DataLoader(val_dataset,batch_size=cf.BATCHSIZE,num_workers=cf.NUM_WORKERS)
    print("Train Num : ",len(train_dataset))
    print("Val Num : ",len(val_dataset))

    #Get model
    net = PortraitNet(num_classes=2).to(device)

    # #Get loss function
    # criterion = nn.CrossEntropyLoss()

    #Get optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=cf.LEARNING_RATE,weight_decay=cf.WEIGHT_DECAY)

    #Get Lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,cf.LR_DECAY_EPOCH,cf.LR_DECAY)

    if args.training_status == 0:
        print("Training from the beginning at epoch 0: ")
        current_epoch = 0
        min_val_loss = 10000
        train_losses = []
        val_losses = []
        val_miou = 0

    elif args.training_status == 1:
        net,optimizer,current_epoch,train_losses,val_losses,val_miou = load_checkpoint(net,optimizer,args.output_folder,args.pretrained_model,device=device)
        print("Training from with latest model at epoch : {} , number epoch for continue training : {}".format(current_epoch,args.epochs))

    elif args.training_status == 2:
        net,optimizer,current_epoch,train_losses,val_losses,val_miou = load_checkpoint(net,optimizer,args.output_folder,args.pretrained_model,device=device)
        print("Training from with best model at epoch : {}, number epoch for continue training : {}".format(current_epoch,args.epochs))

    else:
        raise ValueError("training_status param should be 0/1/2")
    
    #Get optimizer
    for g in optimizer.param_groups:
        g['lr']=1e-3
    
    best_val_iou = 0
    for epoch in range(current_epoch+1,current_epoch+1+args.epochs):
        print("Epoch : {}/{} , learning rate : {}".format(epoch,current_epoch+args.epochs,optimizer.param_groups[0]['lr']))
        train_loss = train(train_loader,net,optimizer,device)
        val_loss,val_iou = test(val_loader,net,device,num_classes=2)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        checkpoints = {
            "state_dict": net.state_dict(),
            "trained_num_epochs": epoch,
            "optimizer": optimizer.state_dict(),
            "train_losses":train_losses,
            "val_losses": val_losses,
            "val_iou": val_iou,
        }
        #Save best model
        if best_val_iou < val_iou:

            best_val_iou = val_iou
            save_checkpoint(checkpoints,args.output_folder,args.model_name)

        save_checkpoint(checkpoints,args.output_folder,"latest.pth.tar")
        print("Train Loss : {}, Val Loss : {} , Val mIOU : {} , best Val mIOU : {}".format(train_loss,val_loss,val_iou,best_val_iou))
        lr_scheduler.step()





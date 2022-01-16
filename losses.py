import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_loss(student_outputs, teacher_outputs, T):
    loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T,dim=1), F.softmax(teacher_outputs/T,dim=1))*T*T
    return loss
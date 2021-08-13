import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.09):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing 

    def forward(self, logits, labels):
        labels[labels == 1] = 1 - self.smoothing 
        labels[labels == 0] = self.smoothing 
        return F.binary_cross_entropy_with_logits(logits, labels)
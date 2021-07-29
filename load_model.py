import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import timm

from dataset_test import SETIdataset
from pathlib import Path


def evaluate(data_loader):
    model.eval()
    
    final_targets = []
    final_outputs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            output = model(inputs)
            
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(output)
            
    return final_outputs, final_targets


device = torch.device('cuda')
model = timm.create_model('resnet50', pretrained=True, in_chans=1, num_classes=1).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('./checkpoint/best_ckpt.pth'))

submission = pd.read_csv('./sample_submission.csv')
path = Path('/home/datasets/SETI/')

test_dataset = SETIdataset(submission, path)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=4)


predictions, valid_targets = evaluate(test_loader)

predictions = np.array(predictions)

predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

submission.target = predictions

submission.to_csv('submission.csv', index=False)
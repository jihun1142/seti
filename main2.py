'''Train CIFAR10 with PyTorch.'''
from typing import List
import torch #torch 모듈을 가져옴
import torch.nn as nn # 신경망 생성하기위해 사용
import torch.optim as optim # 신경망 최적화 알고리즘을 가진 패키지(step이란 매서드를 사용해 각 매개변수를 자동으로 업데이트)
import torch.nn.functional as F #nn모듈의 활성화 함수, 손실함수 등과 같은 매서드를 가져옴
# 관례적으로 F에 임포트 한다.
import torch.backends.cudnn as cudnn 
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomResizedCrop, ToPILImage

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
	Compose, ShiftScaleRotate, Blur, Resize, Cutout
)

import timm
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os # os(operating system) 운영체제를 제어할 수 있도록 해줌.
import argparse

from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar # utils.py에서 progress_bar 함수를 불러옴.
from dataset import SETIdataset
from pathlib import Path
from datetime import datetime
from gridmask import GridMask
from main3 import LabelSmoothing
from torchsampler import ImbalancedDatasetSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--img', default=512, type=int, help='image size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
args = parser.parse_args() #argparse를 쓰려면 위와 같은 코드가 기본적으로 필요함.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_score = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = A.Compose([
    #ShiftScaleRotate(rotate_limit=0),
    Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ShiftScaleRotate(p= 0.5, shift_limit= 0.2, scale_limit= 0.2,rotate_limit=20, border_mode= 0, value= 0, mask_value= 0),
    A.RandomResizedCrop(p=1.0, scale=[0.9,1.0], height=512, width=512),
    # A.Cutout(
    #     num_holes=10, max_h_size=12, max_w_size=12,
    #     fill_value=0, always_apply=False, p=0.5),
    # A.OneOf([
    # GridMask(num_grid=3, rotate=15),
    # GridMask(num_grid=(3,7)),
    # GridMask(num_grid=3,mode=2)
    # ],p=1),
    ToTensorV2()
])
transform_test = A.Compose([
    Resize(512, 512),
    ToTensorV2()
])

# transform_train = Compose([
# 	A.Resize(512, 512),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Rotate(limit=180, p=0.5),
#     A.RandomBrightness(limit=0.6, p=0.5),
#     A.Cutout(
#         num_holes=10, max_h_size=12, max_w_size=12,
#         fill_value=0, always_apply=False, p=0.5),
#     A.ShiftScaleRotate(
#             shift_limit=0.25, scale_limit=0.1, rotate_limit=0),
#     ToTensorV2(p=1.0)

# ])

# transform_train = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((args.img, args.img)),
#     transforms.ToTensor()
# ])

# transform_test = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((args.img, args.img)),
#     transforms.ToTensor()
# ])

# transform_test = Compose([
#     Resize(512, 512),
#    	ToTensorV2(p=1.0)
# ])



path = Path('/home/datasets/SETI/')
csv_file = path.joinpath('train_labels.csv')

df = pd.read_csv(csv_file)
df = sklearn.utils.shuffle(df)
skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# print(skf.split(df, df.target))
for train_idx, test_idx in skf.split(df, df.target):
    print(train_idx)
    print(test_idx)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print(len(train_df))
    print(len(test_df))
    break

#criterion = LabelSmoothing()
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
trainset = SETIdataset(df=train_df, path=path, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, sampler=ImbalancedDatasetSampler(trainset), num_workers=4)

testset = SETIdataset(df=test_df, path=path, transform=transform_test)
testloader = torch.utils.data.DataLoader(
   testset, batch_size=128, shuffle=False, num_workers=4) # 데이터를 불러옴.


# Model
print('==> Building model..')
net = timm.create_model('resnet18d', pretrained=True, in_chans=1, num_classes=1)
net = net.to(device) 
if device == 'cuda':
    net = torch.nn.DataParallel(net) 

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1.0e-7, last_epoch=-1, verbose=True)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[11,23], gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=19, T_mult= 1, eta_min=1e-7, last_epoch=-1, verbose=True)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, pct_start=0.111, anneal_strategy='cos', div_factor=1.0e+2, final_div_factor=1)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join(f'runs/{current_time}')
writer = SummaryWriter(log_dir)
# Training
def train():
    net.train()
    train_loss = 0
    all_targets = []
    all_predictions = [] 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        all_targets.extend(targets.cpu().detach().numpy().astype(int).tolist())
        all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())

        roc_auc = roc_auc_score(all_targets, all_predictions) 
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | score: %.3f '
                     % (train_loss/(batch_idx+1), roc_auc))
    return train_loss/(batch_idx+1), roc_auc

def test():
    global best_score
    net.eval()
    test_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            all_targets.extend(targets.cpu().detach().numpy().tolist())
            all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())
            roc_auc = roc_auc_score(all_targets, all_predictions)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | score: %.3f '
                         % (test_loss/(batch_idx+1), roc_auc))

       # Save checkpoint.
    if roc_auc > best_score:
        print('Saving..')
        torch.save(net.state_dict(), os.path.join(log_dir, 'best_ckpt.pth'))
        best_score = roc_auc
    
    return test_loss/(batch_idx+1), roc_auc


for epoch in range(start_epoch, start_epoch+args.epochs):
    print('\n[Epoch: %d]'%epoch)
    train_loss,train_roc_auc=train()
    test_loss, test_roc_auc=test()
    scheduler.step()

    writer.add_scalar('Loss/train',train_loss, epoch)
    writer.add_scalar('Loss/test',test_loss, epoch)
    writer.add_scalar('ROC_AUC/train',train_roc_auc, epoch)
    writer.add_scalar('ROC_AUC/test',test_roc_auc, epoch)


print('Saving..')
torch.save(net.state_dict(), os.path.join(log_dir, 'last_ckpt.pth'))
writer.close()
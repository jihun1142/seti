import torch #torch 모듈을 가져옴
import torch.nn as nn # 신경망 생성하기위해 사용
import torch.optim as optim # 신경망 최적화 알고리즘을 가진 패키지(step이란 매서드를 사용해 각 매개변수를 자동으로 업데이트)
import torch.nn.functional as F #nn모듈의 활성화 함수, 손실함수 등과 같은 매서드를 가져옴
# 관례적으로 F에 임포트 한다.
import torch.backends.cudnn as cudnn 
import numpy as np

import torchvision #객체를 미세하게 조정하여 검출할 수 있도록 하는 모듈
import torchvision.transforms as transforms # 다양한 이미지 변환 기능을 제공, 데이터를 불러오면서 바로 전처리를 할 수 있게 해준다.
import torchvision.models as models

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
	Compose, ShiftScaleRotate, Blur, Resize, Cutout
)


import timm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os # os(operating system) 운영체제를 제어할 수 있도록 해줌.
import argparse

from torchvision.transforms.transforms import ToPILImage #parse 객체를 생성할 수 있도록 해줌.

from utils import progress_bar # utils.py에서 progress_bar 함수를 불러옴.
from dataset import SETIdataset
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--img', default=224, type=int, help='image size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
args = parser.parse_args() #argparse를 쓰려면 위와 같은 코드가 기본적으로 필요함.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_score = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_test = A.Compose([
    Resize(512, 512),
    ToTensorV2()
])

path = Path('/home/datasets/SETI/')
df = pd.read_csv('./sample_submission.csv')
testset = SETIdataset(df=df, path=path, transform=transform_test, is_train=False)
testloader = torch.utils.data.DataLoader(
   testset, batch_size=128, shuffle=False, num_workers=4) # 데이터를 불러옴.


# Model
print('==> Building model..')
net = timm.create_model('resnet18d', pretrained=True, in_chans=1, num_classes=1)
net = net.to(device) 
if device == 'cuda':
    net = torch.nn.DataParallel(net) 
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/best_ckpt.pth')
net.load_state_dict(checkpoint)


def test():
    net.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            
            all_predictions.extend(outputs.cpu().detach().numpy().tolist())

            progress_bar(batch_idx, len(testloader))
    return all_predictions


all_prediction = test()
predictions = np.array(all_prediction)
predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
df.target = predictions
df.to_csv('submission.csv', index=False)
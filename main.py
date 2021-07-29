'''Train CIFAR10 with PyTorch.'''
import torch #torch 모듈을 가져옴
import torch.nn as nn # 신경망 생성하기위해 사용
import torch.optim as optim # 신경망 최적화 알고리즘을 가진 패키지(step이란 매서드를 사용해 각 매개변수를 자동으로 업데이트)
import torch.nn.functional as F #nn모듈의 활성화 함수, 손실함수 등과 같은 매서드를 가져옴
# 관례적으로 F에 임포트 한다.
import torch.backends.cudnn as cudnn 

import torchvision #객체를 미세하게 조정하여 검출할 수 있도록 하는 모듈
import torchvision.transforms as transforms # 다양한 이미지 변환 기능을 제공, 데이터를 불러오면서 바로 전처리를 할 수 있게 해준다.
import torchvision.models as models

import timm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os # os(operating system) 운영체제를 제어할 수 있도록 해줌.
import argparse

from torchvision.transforms.transforms import Resize, ToPILImage #parse 객체를 생성할 수 있도록 해줌.

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
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img, args.img)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img, args.img)),
    transforms.ToTensor()
]) # 데이터의 전처리를 설정.


path = Path('/home/datasets/SETI/')
csv_file = path.joinpath('train_labels.csv')

df = pd.read_csv(csv_file)
train_df, test_df = train_test_split(df, test_size=0.2)
trainset = SETIdataset(df=train_df, path=path, transform=transform_train)
print(trainset[0][0].shape)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = SETIdataset(df=test_df, path=path, transform=transform_test)
testloader = torch.utils.data.DataLoader(
   testset, batch_size=100, shuffle=False, num_workers=4) # 데이터를 불러옴.


# Model
print('==> Building model..')
net = timm.create_model('resnet50', pretrained=True, in_chans=1, num_classes=1)
net = net.to(device) 
if device == 'cuda':
    net = torch.nn.DataParallel(net) 

criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.5, weight_decay=5e-4) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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
        all_targets.extend(targets.cpu().detach().numpy().tolist())
        all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())
        roc_auc = roc_auc_score(all_targets, all_predictions) 
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | score: %.3f '
                     % (train_loss/(batch_idx+1), roc_auc))

def test(epoch):
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
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/best_ckpt.pth')
        best_score = roc_auc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()


print('Saving..')
torch.save(net.state_dict(), './checkpoint/last_ckpt.pth')

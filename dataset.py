from torch.utils.data import Dataset

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

class SETIdataset(Dataset):
    def __init__(self, df:pd.DataFrame, path:str, transform=None):
        self.df = df
        self.targets = self.df['target'].values
        self.id = self.df['id'].values
        self.path = path
        self.transform = transform

    def __getitem__(self, index: int):
        label = torch.FloatTensor([self.targets[index]])
        filename = self.id[index]

        npy_file = self.path.joinpath('train', filename[0], filename+'.npy')
        img = np.load(npy_file).astype(np.float32)
        img = np.vstack(img).transpose(1,0)

        if self.transform:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.df)

if __name__ =="__main__":
    path = Path('/home/datasets/SETI/')
    csv_file = path.joinpath('train_labels.csv')

    df = pd.read_csv(csv_file)
    trainset, testset = train_test_split(df, test_size=0.2)

    dataset = SETIdataset(trainset, path)
    img, label = dataset[0]
    print(img.shape)
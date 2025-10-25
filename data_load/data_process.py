import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
import scipy.io
import os

class Dataset_Dpos(Dataset):
    def __init__(self, data_dir, target_channel=0, L=12, S=1,
                 mode='train', normalize=True):
        """
        mode: 'train', 'val', 'test1', 'test2'
        """

        # 1. 加载 .mat 数据
        path_l1 = os.path.join(data_dir, 'l1.mat')
        path_l2 = os.path.join(data_dir, 'l2.mat')
        path_l3 = os.path.join(data_dir, 'l3.mat')

        # 加载每个 mat 文件（直接是 np.ndarray）
        l1 = scipy.io.loadmat(path_l1)['data']  # [T, 6]
        l2 = scipy.io.loadmat(path_l2)['data']
        l3 = scipy.io.loadmat(path_l3)['data']
        
        assert l1.shape == l2.shape == l3.shape, "三个节点数据维度不一致！"
        
        T = l1.shape[0]

        # 2. 选定目标通道
        input_seq = np.stack([l2[:, target_channel], l3[:, target_channel]], axis=-1)  # [T, 2]
        target_seq = l1[:, target_channel]  # [T]

        # 3. 设置不同模式的时间区间
        index_range = {
            'val':  (13999, 14999),
            'train':  (10699, 13999),
            'test1':  (1299,  5299),
            'test2':  (6599,  9899),
        }

        if mode not in index_range:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(index_range.keys())}")

        start_idx, end_idx = index_range[mode]
        self.X, self.Y = [], []
        self.x_mark, self.y_mark = [], []

        for t in range(start_idx + L, end_idx - S + 1):
            x_t = input_seq[t - L:t, :]            # [L, 2]
            y_t = target_seq[t + S - 1]            # scalar

            # 相对时间编码 [0, ..., L+S-1] / (L+S)
            full_range = np.arange(L + S) / (L + S)
            self.x_mark.append(full_range[:L].reshape(L, 1))    # [L, 1]
            self.y_mark.append(full_range[L:].reshape(S, 1))    # [S, 1]

            self.X.append(x_t)
            self.Y.append([y_t])  # [1]

        self.X = np.stack(self.X)        # [B, L, 2]
        self.Y = np.stack(self.Y)        # [B, 1]
        self.x_mark = np.stack(self.x_mark)  # [B, L, 1]
        self.y_mark = np.stack(self.y_mark)  # [B, S, 1]

        # 标准化
        self.normalize = normalize
        if normalize and mode == 'train':
            self.mean = self.X.reshape(-1, 2).mean(0, keepdims=True)
            self.std = self.X.reshape(-1, 2).std(0, keepdims=True) + 1e-6
            self.scale = (self.mean, self.std) 
        elif normalize and mode != 'train':
            raise ValueError("必须先初始化 train 集合来获取标准化参数")

    def set_norm_stats(self, mean, std):
        self.mean = mean
        self.std = std
        self.scale = (self.mean, self.std) 
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()         # [L, 2]
        y = torch.from_numpy(self.Y[idx]).float()         # [1]
        x_mark = torch.from_numpy(self.x_mark[idx]).float()  # [L, 1]
        y_mark = torch.from_numpy(self.y_mark[idx]).float()  # [S, 1]
        return x, y, x_mark, y_mark


if __name__ == '__main__':
    train_set = Dataset_Dpos('data', target_channel=0, L=12, S=1, mode='train', normalize=True)
    val_set = Dataset_Dpos('data', target_channel=0, L=12, S=1, mode='val', normalize=False)
    val_set.set_norm_stats(train_set.mean, train_set.std)

    x, y, x_mark, y_mark = train_set[0]
    print("X:", x.shape)         # [12, 2]
    print("Y:", y.shape)         # [1]
    print("x_mark:", x_mark.shape)  # [12, 1]
    print("y_mark:", y_mark.shape)  # [1, 1]

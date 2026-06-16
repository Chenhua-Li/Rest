import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tasks.base import Task


class AddingTaskDataset(Dataset):
    def __init__(self, X, Y, num_classes):
        assert X.dtype == bool
        assert Y.dtype == np.int8
        self.X = torch.FloatTensor(X.astype(np.int8))
        self.Y = torch.IntTensor(Y)
        self.num_classes = num_classes

    def check(self):
        count = [0] * self.num_classes
        for i in range(len(self)):
            assert torch.allclose(torch.cumsum(self.X[i, :, 0] * self.X[i, :, 1], dim=0), self.Y[i])
            count[self.Y[i, -1]] += 1
        print(count)
            
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AddingTask(Task):
    def __init__(self, root='./data/adding_task', seq_len=500, num_classes=10, N_train=10000, N_test=2000, force_regenerate=False):
        self.root = root
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.N_train = N_train
        self.N_test = N_test

        path = os.path.join(root, f'seq_{seq_len}_num_{num_classes}_N_{N_train}_{N_test}.npz')

        if os.path.exists(path) and not force_regenerate:
            data = np.load(path)
            train_X = data['X_train']
            train_Y = data['Y_train']
            test_X = data['X_test']
            test_Y = data['Y_test']
        else:
            train_X, train_Y, test_X, test_Y = AddingTask.generate_data(N_train, N_test, seq_len, num_classes)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, X_train=train_X, Y_train=train_Y, X_test=test_X, Y_test=test_Y)

        self.train_dataset = AddingTaskDataset(train_X, train_Y, num_classes)
        self.test_dataset = AddingTaskDataset(test_X, test_Y, num_classes)
    
    def get_time_window(self):
        return self.seq_len
    
    def has_label_each_step(self):
        return True
    
    def preprocess_data(self, input, label):
        return input.permute(1, 0, 2).float(), label.permute(1, 0).long()

    @staticmethod
    def combine(N, seq_len, vec, pos):
        X = np.zeros((2, N, seq_len), dtype=int)
        Y = np.zeros((N, seq_len), dtype=int)
        X[0] = vec
        for i in range(N):
            X[1, i, pos[i]] = 1
            Y[i] = np.cumsum(X[0, i] * X[1, i], dtype=int)
        X = np.transpose(X, (1, 2, 0))
        return X, Y

    @staticmethod
    def generate_data(N_train, N_test, seq_len, num_classes):
        rng = np.random.default_rng(seed=42)

        N = N_train + N_test
        N_ = int(N * 2)
        vec = rng.integers(2, size=(N_, seq_len), dtype=int)
        pos = np.zeros((N_, num_classes-1), dtype=int)
        ret = np.zeros(N_, dtype=int)
        indices = [[] for _ in range(num_classes)]
        ret_desired = rng.integers(num_classes, size=N_, dtype=int)
        for i in range(N_):
            p = np.where(vec[i] == 1, ret_desired[i], num_classes - 1 - ret_desired[i])
            p = p / np.sum(p)
            pos[i] = rng.choice(seq_len, num_classes-1, replace=False, p=p)
            ret[i] = np.sum(vec[i, pos[i]], dtype=int)
            indices[ret[i]].append(i)

        train_num_per_class = (N_train - 1) // num_classes + 1
        test_num_per_class = (N_test - 1) // num_classes + 1

        train_nums = np.array([train_num_per_class] * num_classes)
        test_nums = np.array([test_num_per_class] * num_classes)
        if N_train % num_classes != 0:
            train_nums[-(train_num_per_class * num_classes - N_train):] -= 1
        if N_test % num_classes != 0:
            test_nums[-(test_num_per_class * num_classes - N_test):] -= 1
        assert np.sum(train_nums) == N_train
        assert np.sum(test_nums) == N_test
        
        # print([len(indices[i]) for i in range(num_classes)])
        for i in range(num_classes):
            assert len(indices[i]) >= train_nums[i] + test_nums[i], 'Not enough samples for class %d' % i

        for i in range(num_classes):
            indices[i] = np.array(indices[i])
            rng.shuffle(indices[i])
        
        train_indices = np.concatenate([indices[i][:train_nums[i]] for i in range(num_classes)])
        test_indices = np.concatenate([indices[i][train_nums[i]:train_nums[i] + test_nums[i]] for i in range(num_classes)])

        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        train_X, train_Y = AddingTask.combine(N_train, seq_len, vec[train_indices], pos[train_indices])
        test_X, test_Y = AddingTask.combine(N_test, seq_len, vec[test_indices], pos[test_indices])

        train_X = train_X.astype(bool)
        test_X = test_X.astype(bool)
        train_Y = train_Y.astype(np.int8)
        test_Y = test_Y.astype(np.int8)
        
        return train_X, train_Y, test_X, test_Y

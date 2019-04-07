import numpy as np
import torch
from torch.utils.data.dataset import Dataset
class FrameDataset(Dataset):
    def __init__(self, num_paddings, X, y=None):
        self.__xs = []
        self.__ys = []
        self.num_paddings = num_paddings
        padder = np.array([[0]*len(X[0][0])]*num_paddings)
        for utterance in X:
            padded = np.append(np.append(padder, utterance, axis=0), padder, axis=0)
            self.__xs.extend([np.hstack(padded[i-num_paddings:i+num_paddings+1]).tolist() for i in range(num_paddings,len(padded)-num_paddings,1)])
        del X
        if y is not None :
            [self.__ys.extend(utterance.tolist()) for utterance in y]
            del y
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        
        instance = torch.from_numpy(np.array(self.__xs[index])).float()
        if len(self.__ys) != 0:
            label = torch.from_numpy(np.array(self.__ys[index]))
            return instance, label
        return instance

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)
    
class FrameDataset_small(Dataset):
    def __init__(self, num_paddings, X, y=None):
        self.__xs = []
        self.__ys = []
        self.num_paddings = num_paddings
        self.checker = [0]
        self.size = 0
        self.padder = np.array([[0]*len(X[0][0])]*num_paddings)
        for utterance in X:
            padded = np.append(np.append(self.padder, utterance, axis=0), self.padder, axis=0)
            self.__xs.append(padded)
            self.checker.append(self.checker[-1]+len(utterance))
            self.size += len(utterance)
        del X
        if y is not None :
            [self.__ys.extend(utterance.tolist()) for utterance in y]
            del y
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # utterance_index = 0
        # pointer = 0
        for idx, point in enumerate(self.checker):
            if point <= index:
                # utterance_index = idx
                # pointer = point
                pass
            else:
                break
        utterance = self.__xs[idx-1]
        local_index = index-self.checker[idx-1]+self.num_paddings
        instance = torch.from_numpy(utterance[local_index-self.num_paddings:local_index+self.num_paddings+1]).float()
        if len(self.__ys) != 0:
            label = torch.from_numpy(np.array(self.__ys[index]))
            return instance, label
        return instance

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.size

class FrameDataset_quick(Dataset):
    def __init__(self, data, dataset_index_file, num_padding, test_mode=False):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.num_padding = num_padding
#         self.X = []
        holder = []
        self.padder = np.array([[0]*len(data[0][0])]*self.num_padding)
        self.checker = [0]
        self.size = 0
        for utterance in data:
            padded = torch.from_numpy(np.append(np.append(self.padder, utterance, axis=0), self.padder, axis=0)).float()
#             self.X.append(padded.tolist())
            holder.append(padded)
            self.checker.append(self.checker[-1]+len(utterance))
            self.size += len(utterance)
        self.X = torch.cat(holder)
        del data, holder
        self.dataset_index = open(dataset_index_file).readlines()
        self.test = test_mode

    def __getitem__(self, index):
        x, y, label = self.dataset_index[index].split(' ')
        x, y = int(x), int(y)
        real_x = self.checker[x]+y+self.num_padding*(1+x)
#         instance = np.array(self.X[x])[y:y+self.num_padding*2+1].flatten()
        instance = self.X[real_x-self.num_padding:real_x+self.num_padding+1]
        if self.test:
            return instance.float()
        label_torch = torch.from_numpy(np.array(int(label)))
        return instance.float(), label_torch

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.size

class FrameDataset_revise(Dataset):
    def __init__(self, data, dataset_index_file, num_padding, test_mode=False):
        self.data = data
        self.dataset_index = open(dataset_index_file).readlines()
        self.test = test_mode
        self.num_padding = num_padding
        self.size = sum([len(utterance) for utterance in data])

    def __getitem__(self, index):
        x, y, label = self.dataset_index[index].split(' ')
        x, y = int(x), int(y)
        index = np.clip(np.arange(y-self.num_padding, y+self.num_padding+1), 0, len(self.data[x])-1).astype(int)
        instance = torch.from_numpy(self.data[x][(index)])
        if self.test:
            return instance.float()
        label_torch = torch.from_numpy(np.array(int(label)))
        return instance.float(), label_torch

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.size
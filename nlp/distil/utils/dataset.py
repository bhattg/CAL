import torch
from torch.utils.data import Dataset

class MedMNISTDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert tensors[0].shape[0] == tensors[1].shape[0]
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.tensors[1][index])
        
        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]
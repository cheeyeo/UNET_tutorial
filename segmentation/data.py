from collections import namedtuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
])

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

Label = namedtuple("Label", ["name", "train_id", "color"])

drivables = [ 
    Label("direct", 0, (219, 94, 86)),        # red
    Label("alternative", 1, (86, 211, 219)),  # cyan
    Label("background", 2, (0, 0, 0)),        # black          
]

train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)


class BDD100K_dataset(Dataset):
    def __init__(self, images, labels, tf):
        # self.tf is the transform
        super(BDD100K_dataset, self).__init__()
        self.images = images
        self.labels = labels
        self.tf = tf
    
    
    def __len__(self):
        return self.images.shape[0]
    

    def __getitem__(self, index):
        rgb_image = self.images[index]
        if self.tf is not None:
            rgb_image = self.tf(rgb_image)
        
        label_image = torch.from_numpy(self.labels[index]).long()
        return rgb_image, label_image


def get_datasets(images, labels):
    data = BDD100K_dataset(images, labels, tf=preprocess)
    total_count = len(data)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_set, val_set, test_set = torch.utils.data.random_split(data, (train_count, valid_count, test_count), generator=torch.Generator().manual_seed(1))
    return train_set, val_set, test_set


def get_dataloaders(train_set, val_set, test_set):
    train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_set, batch_size=8)
    test_dataloader = DataLoader(test_set, batch_size=8)
    return train_dataloader, val_dataloader, test_dataloader
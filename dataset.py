
import csv
from enum import Enum
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import option

class Metadata(Enum):
    NAME = 0
    IMAGE = 1
    MD5 = 2
    DATASET = 3
    SPLIT = 4

class CelebHQ(Dataset):
    def __init__(self, args, state, transforms):
        self.csv_file_path = args.csv_file_path
        self.dataset = args.dataset
        self.state = state
        self._parse_list()
        self.transforms = transforms

    def _parse_list(self):
        count = 0
        self.data = []
        with open(self.csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader.reader:
                if row[0] == 'name':
                    continue
                if self.state == 0 and row[Metadata.SPLIT.value] == 'train':
                    self.data.append(row)
                    count += 1
                if self.state == 1 and row[Metadata.SPLIT.value] == 'test':
                    self.data.append(row)
                    count += 1

        self.num = count

    def __getitem__(self, index):
        meta = self.data[index]
        name = meta[Metadata.DATASET.value] + '_' + meta[Metadata.NAME.value]
        image = Image.open(meta[Metadata.IMAGE.value])#.convert("RGB")
        image = self.transforms(image)
        return name, image

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    args = option.parser.parse_args()
    train_dataset = CHDD(args, None)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    for i, (name, image) in enumerate(train_loader):
        print(name, image.shape)

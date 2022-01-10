import random
import os

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets.folder import default_loader


class Food_5K_dataset(Dataset):
    def __init__ (self, root, transform = None, phase = 'train', seed = 0):
        self.classes, self.samples = self.read_dir(root, seed)
        self.loader = default_loader
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

        # print('classes : \n', self.classes)
        # print('# of samples : \n', len(self.samples))
        # print('samples : \n', self.samples)


    def read_dir (self, root, seed = 0):
        img_ext = ['.jpg', '.png']
        random.seed(seed)

        classes = []
        samples = []

        for root, dirs, files in os.walk(root):
            dirs.sort(key=str.lower)
            img_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in img_ext, files))
            if img_files:
                random.shuffle(img_files)
                samples += [(os.path.join(root, f), len(classes)) for f in img_files]
                classes.append(os.path.basename(root))
            # else: # TODO
            #     classes.append('unknown')

        return classes, samples


    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.transform(self.loader(path))

        return target, image, path


    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from binary_classification.autoaugment import ImageNetPolicy

    transform = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    }

    train_loader = DataLoader(
        Food_5K_dataset('/hdd/Food-5K/training/', transform=transform['train'], phase='train'), batch_size=128,
        num_workers=4, shuffle=True)

    valid_loader = DataLoader(
        Food_5K_dataset('/hdd/Food-5K/validation/', transform=transform['valid'], phase='valid'),
        batch_size=128,
        num_workers=4, shuffle=True)
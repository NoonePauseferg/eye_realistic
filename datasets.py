from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

class Eye_dataset(Dataset):
    """zip save the world"""
    def __init__(self, X, y, transform):
        super().__init__()
        self.fake_imgs = X[:y.shape[0]]
        self.real_imgs = y
        self.transform = transform

    def __getitem__(self, index):
        real = self.real_imgs[index]
        fake = self.fake_imgs[index]
        if self.transform:
            real = self.transform(real)
            fake = self.transform(fake)
        return real, fake

    def __len__(self):
        return self.real_imgs.shape[0]

class lightning_EyeDatset(pl.LightningDataModule):
    def __init__(self, Xy_train, Xy_val):
        super().__init__()
        self.train = Xy_train
        self.val = Xy_val
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def setup(self, stage : str):
        if stage == "fit":
            self.eye_train = Eye_dataset(*self.train, self.train_transforms)
            self.eye_validate = Eye_dataset(*self.val, self.val_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.eye_train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.eye_validate, batch_size=128)
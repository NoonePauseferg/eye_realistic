#https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
#https://www.youtube.com/watch?v=_pIMdDWK5sc
from models import Refiner_, Discriminator_
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision.utils import make_grid
import torch
from torchvision import transforms
from datasets import Eye_dataset, lightning_EyeDatset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class SimGAN(pl.LightningModule):
    def __init__(self, 
                 train_data, 
                 val_data, 
                 hparams={
                    'lr'          : 0.001,
                    'ref_lambda'  : 0.3,
                    'num_workers' : 2
                 }):
        super().__init__()
        self.hparams.update(hparams)
        self.refiner = Refiner_(10)
        self.discriminator = Discriminator_()
        self.img_buffer = None
        self.epoch = 0
        self.val_step = 0
        self.train_step = 0
        self.train_data = train_data
        self.val_data = val_data
        self.buffer = None

    def forward(self, x):
        return self.refiner(x)

    def refiner_loss(self, x, y):
        # INPUT [bs, 1, h, w]
        # discriminator_output [bs, 2] : 1 if synth, 0 if real
        # -log(D(K(z))) + ||y - K(z)||_1, x=K(z)

        # d_pred = self.discriminator(x)
        # target = torch.ones((d_pred.shape[0], d_pred.shape[-1]), dtype=torch.long)
        # if self.on_gpu:
        #     target = target.cuda(x.device.index)

        # return F.cross_entropy(d_pred, target) + \
        #         self.hparams.ref_lambda*F.l1_loss(x, y, reduction='sum')

        return self.hparams.ref_lambda*F.l1_loss(x, y, reduction='sum')
    def adversarial_loss(self, x, y):
        return F.cross_entropy(x, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, fake_imgs = batch

        sample_imgs = fake_imgs[:6]
        grid = make_grid(sample_imgs).detach().cpu()
        self.logger.experiment.add_image("generated_images", grid, self.train_step)
        self.train_step += 1

        #refiner
        if optimizer_idx == 0:
            # INPUT [bs, 1, h, w]
            refiner_imgs = self(fake_imgs)
            refiner_loss = self.refiner_loss(fake_imgs, refiner_imgs)

            output  = OrderedDict({
                'loss' : refiner_loss,
                # 'progress_bar' : tqdm_dict,
            })
            self.log("train_ref_loss", refiner_loss)
            return output

        #discriminator
        if optimizer_idx == 1:
            #----REAl----
            real_predict = self.discriminator(real_imgs)
            valid = torch.zeros((real_predict.shape[0], real_predict.shape[-1]), dtype=torch.long)
            if self.on_gpu:
                valid = valid.cuda(real_imgs.device.index)
            real_loss = self.adversarial_loss(real_predict, valid)

            #----SYNTH----
            #TODO : buffer
            refiner_fake = self(fake_imgs)
            # if not torch.is_tensor(self.buffer):
            #     fake_predict = self.discriminator(refiner_fake)
            #     self.buffer = refiner_fake
            # else:
            #     refiner_data = refiner_fake[np.random.choice(refiner_fake.shape[0], refiner_fake.shape[0]//2)]
            #     buffer_data  = self.buffer[np.random.choice(self.buffer.shape[0], refiner_fake.shape[0]//2)]
            #     fake_predict = self.discriminator(torch.vstack((refiner_data, buffer_data)))
            #     self.buffer[np.random.choice(self.buffer.shape[0], self.buffer.shape[0]//2)] = \
            #         refiner_fake[np.random.choice(refiner_fake.shape[0], self.buffer.shape[0]//2)]
            fake_predict = self.discriminator(refiner_fake)
            fake = torch.ones((fake_predict.shape[0], fake_predict.shape[-1]), dtype=torch.long)
            if self.on_gpu:
                fake = fake.cuda(real_imgs.device.index)
            fake_loss = self.adversarial_loss(fake_predict, fake)

            dis_loss = real_loss + fake_loss
            output  = OrderedDict({
                'loss' : dis_loss,
                # 'progress_bar' : tqdm_dict,
            })
            self.log("train_dis_loss", dis_loss)
            return output

    def validation_step(self, batch, batch_idx):
        real_imgs, fake_imgs = batch
        # sample_imgs = fake_imgs[:6]
        # grid = make_grid(sample_imgs).detach().cpu()
        # self.logger.experiment.add_image("generated_images", grid, self.val_step)
        # self.val_step += 1

        real_pred = self.discriminator(real_imgs)
        # [128, 2, 36]
        valid = torch.zeros((real_pred.shape[0], real_pred.shape[-1]), dtype=torch.long)
        if self.on_gpu:
            valid = valid.cuda(real_imgs.device.index)
        real_loss = self.adversarial_loss(real_pred, valid)

        refiner_imgs = self(fake_imgs)
        refiner_loss = self.refiner_loss(fake_imgs, refiner_imgs)
        fake_pred = self.discriminator(fake_imgs)
        fake = torch.ones((fake_pred.shape[0], fake_pred.shape[-1]), dtype=torch.long)
        if self.on_gpu:
            fake = fake.cuda(fake_imgs.device.index)
        fake_loss = self.adversarial_loss(fake_pred, fake)
        
        dis_loss = real_loss + fake_loss
        
        

        tqdm_dict = {
            'val_dis_loss' : dis_loss,
            'val_ref_loss' : refiner_loss
        }

        output = OrderedDict({
            'val_loss' : dis_loss,
            # 'progress_bar' : tqdm_dict,
        })
        self.log("val_ref_loss", refiner_loss)
        self.log("val_dis_loss", dis_loss)
        return output
    
    def validation_epoch_end(self, outputs):
        self.epoch += 1
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {
            'val_loss' : avg_loss,
        }
        output = OrderedDict({
            'val_loss' : avg_loss,
            'log' : log
        })
        return output


    def configure_optimizers(self):
        lr = self.hparams["lr"]
        refinder_optim      = torch.optim.Adam(self.refiner.parameters(), lr=lr, betas=(0.9, 0.999))
        discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9,0.999))
        return [refinder_optim, discriminator_optim], []

    # def train_dataloader(self):
    #     train_transforms = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(10),
    #     ])

    #     train_dataset = Eye_dataset(*self.train_data, train_transforms)

    #     return DataLoader(train_dataset, 
    #                       batch_size=128,
    #                       shuffle=True, 
    #                       num_workers=10)

    # def val_dataloader(self):
    #     val_transforms = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         transforms.ToTensor(),
    #     ])

    #     val_dataset = Eye_dataset(*self.val_data, val_transforms)

    #     return DataLoader(val_dataset,
    #                       batch_size=128,
    #                       shuffle=False, 
    #                       num_workers=10)


if __name__ == "__main__":
# X_train.shape,     Y_train.shape,    X_test.shape,      Y_test.shap
#(40000, 35, 55, 1) (3200, 35, 55, 3) (10000, 35, 55, 1) (800, 35, 55, 3)
    X_train = np.random.randn(1024, 35, 55, 1).astype(np.float32)
    X_test = np.random.randn(1024, 35, 55, 1).astype(np.float32)
    Y_train = np.random.randn(1024, 35, 55, 3).astype(np.float32)
    Y_test = np.random.randn(1024, 35, 55, 3).astype(np.float32)
    sgn = SimGAN((X_train, Y_train), (X_test, Y_test))
    trainer = pl.Trainer(max_epochs=1,
                         max_steps=10,
                         )
    trainer.fit(sgn)
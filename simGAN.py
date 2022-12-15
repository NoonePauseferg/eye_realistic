#https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
#https://www.youtube.com/watch?v=_pIMdDWK5sc
from collab.models import Refiner_, Discriminator_
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision.utils import make_grid
import torch
from torchvision import transforms
from collab.datasets import Eye_dataset, lightning_EyeDatset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class SimGAN(pl.LightningModule):
    def __init__(self, 
                 train_data, 
                 val_data, 
                 hparams={
                    'lr'          : 0.001,
                    'ref_lambda'  : 0.0001,
                    'log_weight'  : 20,
                    'num_workers' : 2,
                    'fake_loss_weight' : 5
                 }):
        super().__init__()
        self.hparams.update(hparams)
        self.refiner = Refiner_(4)
        self.discriminator = Discriminator_()
        self.img_buffer = None
        self.epoch = 0
        self.valid_step = 0
        self.train_step = 0
        self.train_data = train_data
        self.val_data = val_data
        self.buffer = None

    def forward(self, x):
        return self.refiner(x)

    def refiner_loss(self, x, y, flag = False):
        # INPUT [bs, 1, h, w]
        # discriminator_output [bs, 2] : 1 if synth, 0 if real
        # -log(D(K(z))) + ||y - K(z)||_1, x=K(z)
        d_pred = self.discriminator(x)
        target = torch.zeros((d_pred.shape[0], d_pred.shape[-1]), dtype=torch.long)
        if self.on_gpu:
            target = target.cuda(x.device.index)
        l_real = self.hparams.log_weight * F.cross_entropy(d_pred, target)
        l_reg = self.hparams.ref_lambda * F.l1_loss(x, y, reduction='sum')
        if flag:
            l_real = l_real.detach()
        return l_real + l_reg, l_real, l_reg

        # return self.hparams.ref_lambda*F.l1_loss(x, y, reduction='sum')
        # return self.hparams.log_weight*F.cross_entropy(d_pred, target)
    def adversarial_loss(self, x, y):
        return F.cross_entropy(x, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, fake_imgs = batch
        self.train_step += 1
        flag=False


        total_norm = 0.0
        for p in self.discriminator.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.log("discriminator_weight_norm",total_norm)


        #refiner
        if optimizer_idx == 0:
            # INPUT [bs, 1, h, w]
            refiner_imgs = self(fake_imgs)
            if self.train_step % 2 == 0:
                flag = True
            refiner_loss, l_real, l_reg = self.refiner_loss(refiner_imgs, fake_imgs, flag=flag)

            self.log("l_real", l_real)
            self.log("l_reg", l_reg)
            tqdm_dict = {'ref_loss' : refiner_loss}
            output  = OrderedDict({
                'loss' : refiner_loss,
                'progress_bar' : tqdm_dict,
                'log' : tqdm_dict
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
            refiner_fake = self(fake_imgs).detach()
            # fake_predict = self.discriminator(refiner_fake)

            if not torch.is_tensor(self.buffer):
                fake_predict = self.discriminator(refiner_fake)
                self.buffer = refiner_fake
            else:
                refiner_data = refiner_fake[np.random.choice(refiner_fake.shape[0], refiner_fake.shape[0]//2)]
                buffer_data  = self.buffer[np.random.choice(self.buffer.shape[0], refiner_fake.shape[0]//2)]
                fake_predict = self.discriminator(torch.vstack((refiner_data, buffer_data)))
                self.buffer[np.random.choice(self.buffer.shape[0], self.buffer.shape[0]//2)] = \
                    refiner_fake[np.random.choice(refiner_fake.shape[0], self.buffer.shape[0]//2)]

            fake = torch.ones((fake_predict.shape[0], fake_predict.shape[-1]), dtype=torch.long)
            if self.on_gpu:
                fake = fake.cuda(real_imgs.device.index)
            fake_loss = self.adversarial_loss(fake_predict, fake)

            dis_loss = real_loss + fake_loss
            # dis_loss = fake_loss
            self.log("train_dis_loss", dis_loss)
            tqdm_dict = {'dis_loss' : dis_loss}
            output  = OrderedDict({
                'loss' : dis_loss,
                'progress_bar' : tqdm_dict,
                'log' : tqdm_dict
            })
            return output

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
        ):
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        if optimizer_idx == 1:
            if batch_idx % 2 == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        

    def validation_step(self, batch, batch_idx):
        real_imgs, fake_imgs = batch


        

        real_pred = self.discriminator(real_imgs)
        # [128, 2, 36]
        valid = torch.zeros((real_pred.shape[0], real_pred.shape[-1]), dtype=torch.long)
        if self.on_gpu:
            valid = valid.cuda(real_imgs.device.index)
        real_loss = self.adversarial_loss(real_pred, valid)

        refiner_imgs = self(fake_imgs)
        refiner_loss, _, _ = self.refiner_loss(refiner_imgs, fake_imgs)
        fake_pred = self.discriminator(fake_imgs)
        fake = torch.ones((fake_pred.shape[0], fake_pred.shape[-1]), dtype=torch.long)
        if self.on_gpu:
            fake = fake.cuda(fake_imgs.device.index)
        fake_loss = self.adversarial_loss(fake_pred, fake)
        
        dis_loss = real_loss + self.hparams.fake_loss_weight*fake_loss
        # dis_loss = fake_loss
        
        th = 12
        if self.valid_step % 5 == 0:
            # sample_imgs = torch.clip((real_imgs[:th] + 1.)/2., 0, 1)
            # grid = make_grid(sample_imgs, th//2)
            sample_imgs = refiner_imgs[:th]
            f = lambda x : (x + 1.).detach().cpu() / 2.
            grid = torch.clip(make_grid(torch.vstack((f(sample_imgs), f(fake_imgs[:th]))), th//2), 0, 1)
            self.logger.experiment.add_image("generated_images", grid, self.valid_step)
        self.valid_step+=1

        self.log("val_dis_loss", dis_loss)
        self.log("val_ref_loss", refiner_loss)
        tqdm_dict = {
            'val_dis_loss' : dis_loss,
            'val_ref_loss' : refiner_loss
        }

        output = OrderedDict({
            'val_d_loss' : dis_loss,
            'progress_bar' : tqdm_dict,
            'log' : tqdm_dict
        })
        return output
    
    def configure_optimizers(self):
        lr = self.hparams["lr"]
        refinder_optim      = torch.optim.Adam(self.refiner.parameters(), lr=lr, betas=(0.9, 0.999))
        discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9,0.999))
        return [refinder_optim, discriminator_optim], []


if __name__ == "__main__":
# X_train.shape,     Y_train.shape,    X_test.shape,      Y_test.shap
#(40000, 35, 55, 1) (3200, 35, 55, 3) (10000, 35, 55, 1) (800, 35, 55, 3)
    X_train = np.random.randn(1024, 35, 55, 1).astype(np.float32)
    X_test = np.random.randn(1024, 35, 55, 1).astype(np.float32)
    Y_train = np.random.randn(1024, 35, 55, 3).astype(np.float32)
    Y_test = np.random.randn(1024, 35, 55, 3).astype(np.float32)
    sgn = SimGAN((X_train, Y_train), (X_test, Y_test))
    trainer = pl.Trainer(
        max_epochs=200,
        log_every_n_steps=5,
        accelerator="auto",
        devices=1
    )
    dm = lightning_EyeDatset((X_train, Y_train), (X_test, Y_test))
    trainer.fit(sgn, datamodule=dm)

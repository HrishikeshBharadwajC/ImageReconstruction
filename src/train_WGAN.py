import os, torch, numpy as np, matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
import csv

from discriminator_model_WGAN import Discriminator
from generator_model import Generator
from params import hyperparameters_WGAN as hyperparameters
from dataset import GenerateData
from utils import *
import pandas as pd

class Training():
    def __init__(self, hyperparameters):
        self.num_workers = 2
        self.batch_size = hyperparameters['batch_size']
        self.epochs = hyperparameters['epochs']
        self.critic_iters = hyperparameters['critic_iters']
        self.D_lr = hyperparameters['learning_rate_discriminator']
        self.G_lr = hyperparameters['learning_rate_generator']
        self.checkpoint_iters = hyperparameters['save_every_n_epochs']
        self.dataset_path = hyperparameters['dataset_path']
        self.gp_weight = hyperparameters['gp_weight']
        self.last_checkpoint = hyperparameters['load_model_from_epoch']
        self.G = Generator()
        self.D = Discriminator()
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_direction = hyperparameters['dataset_direction']  
        self.losses = []
        self.G.to(self.device)
        self.D.to(self.device)
    
    def gradient_penalty(self, realImage, Label, FakeImage):
        realImage = realImage.to(self.device)
        FakeImage = FakeImage.to(self.device)
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, Label.size(1), Label.size(2), Label.size(3))
        eta = eta.to(self.device)
        interpolated = eta * Label + ((1 - eta) * FakeImage)
        interpolated = interpolated.to(self.device)
        # define it to calculate gradient
        interpolated = autograd.Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = self.D(realImage, interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, 
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                    create_graph=True, retain_graph=True)[0]
        gradients_debug = gradients + 1e-16

        # grad_penalty = torch.mean(1. - torch.sqrt(1e-8+torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1)))**2
        grad_penalty = ((gradients_debug.norm(2, dim=1) - 1) ** 2).mean() * self.gp_weight
        return grad_penalty

    def provide_batches(self, data_loader):
        while True:
            for i, (RealImages, Labels) in enumerate(data_loader):
                yield RealImages, Labels

    def train(self):
        train_dataset = GenerateData(self.dataset_path + "/train", self.dataset_direction == "reverse")
        val_dataset = GenerateData(self.dataset_path + "/val", self.dataset_direction == "reverse")
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        if self.last_checkpoint > 0:
            self.D, self.optimizerD = load_checkpoint(
                self.dataset_path + f"/checkpoints/{self.last_checkpoint}_discriminator.pth",
                self.D,
                self.optimizerD,
                self.D_lr,
                self.device,
            )
            self.G, self.optimizerG = load_checkpoint(
                self.dataset_path + f"/checkpoints/{self.last_checkpoint}_generator.pth",
                self.G,
                self.optimizerG,
                self.G_lr,
                self.device
            )
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        one.to(self.device)
        mone.to(self.device)
        self.data = self.provide_batches(train_data_loader)

        for gen_epochs in range(self.last_checkpoint, self.epochs):
            for p in self.D.parameters():
                p.requires_grad = True
            self.G.train()
            self.D.train()
            D_loss_real = 0
            D_loss_fake = 0
            D_grad_pen = 0
            total_D_loss = 0
            total_G_loss = 0
            for d_epochs in tqdm(range(self.critic_iters), desc=f"Epoch {gen_epochs+1}"):
                self.D.zero_grad()
                realImage, Label = self.data.__next__()
                realImage = realImage.float().to(self.device)
                Label = Label.float().to(self.device)
                if(realImage.size()[0]!=self.batch_size):
                    continue
                D_loss_real = self.D(realImage, Label)
                D_loss_real = D_loss_real.mean()
                D_loss_real.backward(mone)

                fakeImage = self.G(realImage)
                D_loss_fake = self.D(realImage, fakeImage)
                D_loss_fake = D_loss_fake.mean()
                D_loss_fake.backward(one)

                gradient_penalty = self.gradient_penalty(realImage, Label, fakeImage)
                gradient_penalty.backward()

                d_loss = D_loss_fake - D_loss_real + gradient_penalty
                total_D_loss += d_loss.item()
                self.optimizerD.step()

            for p in self.D.parameters():
                p.requires_grad = False
            self.G.zero_grad()
            fakeImage = self.G(realImage)
            g_loss = self.D(realImage, fakeImage)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            total_G_loss += g_cost.item()
            self.optimizerG.step()
            

            if gen_epochs % self.checkpoint_iters == 0 or gen_epochs == self.epochs - 1:
                print("=> Saving checkpoint")
                save_checkpoint(self.G, self.optimizerG, 1+gen_epochs, self.dataset_path, "generator")
                save_checkpoint(self.D, self.optimizerD, 1+gen_epochs, self.dataset_path, "discriminator")
                save_some_examples(self.G, val_data_loader, gen_epochs+1, self.dataset_path, self.device)
            
            # val_g_loss, val_d_loss = self.evaluate(val_data_loader)
            val_g_loss, val_d_loss =0 ,0
            summary_str = 'Epoch: {}\t Train D Loss: {} \t Train G Loss: {}\t Val D Loss: {} \t Val G Loss: {}'.format(gen_epochs+1, total_D_loss/(self.batch_size*self.critic_iters), total_G_loss/(self.batch_size), val_d_loss, val_g_loss)
            self.__log(summary_str)
            self.losses.append([gen_epochs + 1,total_D_loss/(self.critic_iters*self.batch_size),val_d_loss,total_G_loss/self.batch_size,val_g_loss])
            print(summary_str)
            loss_cols = list(zip(*self.losses))
            plot_curves(self.dataset_path,loss_cols[0],loss_cols[1], ylabel="Discriminator Loss")
            plot_curves(self.dataset_path,loss_cols[0],loss_cols[3], ylabel="Generator Loss")
            pd.DataFrame(loss_cols).to_csv(self.dataset_path+'/losses.csv')

    def __log(self, summary_str, file_name = 'all.log'):
        path = os.path.join(self.dataset_path, file_name)
        with open(path, 'a') as f:
                f.write(summary_str + '\n')
    
    def evaluate(self, val_data_loader):
        val_size = len(val_data_loader.dataset)
        total_d_val_loss = 0.0
        total_g_val_loss = 0.0
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            for idx, (realImage, Label) in enumerate(val_data_loader):
                realImage = realImage.float().to(self.device)
                Label = Label.float().to(self.device)
                # Discriminator
                D_loss_real = self.D(realImage, Label)
                D_loss_real = D_loss_real.mean()
                fakeImage = self.G(realImage)
                D_loss_fake = self.D(realImage, fakeImage)
                D_loss_fake = D_loss_fake.mean()
                d_loss = D_loss_fake - D_loss_real
                total_d_val_loss += d_loss.item()

                g_loss = self.D(realImage, fakeImage)
                g_loss = g_loss.mean()
                g_cost = -g_loss
                total_g_val_loss += g_cost.item()
            

            batch_d_loss = total_d_val_loss/val_size
            batch_g_loss = total_g_val_loss/val_size

            return batch_g_loss, batch_d_loss

if __name__ == '__main__':
    Train_inst = Training(hyperparameters)
    Train_inst.train()



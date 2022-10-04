from discriminator_model import Discriminator
from generator_model import Generator
from dataset import GenerateData

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from config import hyperparameters
from utils import *
import warnings
import numpy as np
warnings.filterwarnings("ignore")
epochs = hyperparameters["epochs"]
lamda = hyperparameters["lamda"]
batch_size = hyperparameters["batch_size"]
num_workers = hyperparameters["num_workers"]
learning_rate_discriminator = hyperparameters["learning_rate_discriminator"]
learning_rate_generator = hyperparameters["learning_rate_generator"]
save_every_n_epochs = hyperparameters["save_every_n_epochs"]
root_path = hyperparameters["dataset_path"]
checkpoint_epoch = hyperparameters["load_model_from_epoch"]
dataset_direction = hyperparameters["dataset_direction"]
train_dataset = GenerateData(root_path + "/train", True if dataset_direction == "reverse" else False)
val_dataset = GenerateData(root_path + "/val", True if dataset_direction == "reverse" else False)
train_data_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
val_data_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator()
generator.to(device)

optimizer_generator = torch.optim.Adam(
    generator.parameters(), lr=learning_rate_generator, betas=(0.5, 0.999)
)
generator_scaler = torch.cuda.amp.GradScaler()

discriminator = Discriminator()
discriminator.to(device)
optimizer_discriminator = torch.optim.Adam(
    discriminator.parameters(), lr=learning_rate_discriminator, betas=(0.5, 0.999)
)
discriminator_scaler = torch.cuda.amp.GradScaler()

if checkpoint_epoch > 0:
    discriminator, optimizer_discriminator = load_checkpoint(
        root_path + f"/checkpoints/{checkpoint_epoch}_discriminator.pth",
        discriminator,
        optimizer_discriminator,
        learning_rate_discriminator,
        device,
    )
    generator, optimizer_generator = load_checkpoint(
        root_path + f"/checkpoints/{checkpoint_epoch}_generator.pth",
        generator,
        optimizer_generator,
        learning_rate_generator,
        device,
    )
    losses = np.loadtxt(root_path + "/all_log.csv", delimiter=",", skiprows=1).tolist()
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss(reduction="mean")
losses = []
train_size = len(train_data_loader.sampler)

def evaluate(discriminator, generator, val_data_loader):
    val_size = len(val_data_loader.sampler)
    total_d_val_loss = 0.0
    total_g_val_loss = 0.0
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_data_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            batch_size = x.size(0)
            # Discriminator
            y_fake = generator(x)
            d_real = discriminator(x, y)
            d_fake = discriminator(x, y_fake)
            d_real_loss = bce_loss(d_real, torch.ones_like(d_real))
            d_fake_loss = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2
            total_d_val_loss += d_loss.item() * batch_size

            # generator
            d_fake = discriminator(x, y_fake)
            g_fake_loss = bce_loss(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(y, y_fake)
            g_loss = g_fake_loss + (l1 * lamda)
            total_g_val_loss += g_loss.item() * batch_size
           
        batch_d_loss = total_d_val_loss / val_size
        batch_g_loss = total_g_val_loss / val_size
        
        return batch_d_loss, batch_g_loss
    
for epoch in range(checkpoint_epoch, epochs):
    total_d_loss = 0.0
    total_g_loss = 0.0
    for idx, (x, y) in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch+1}")):
        x = x.float().to(device)
        y = y.float().to(device)
        batch_size = x.size(0)
        # Discriminator
        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            d_real = discriminator(x, y)
            d_fake = discriminator(x, y_fake.detach())
            d_real_loss = bce_loss(d_real, torch.ones_like(d_real))
            d_fake_loss = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2
            total_d_loss += d_loss.item() * batch_size
            # authors of cycleGAN said to divide the total d_loss by 2 because
            # that would train the discriminator train slower relative to the generator
            # dividing by 2 shouldn't make sense because anyway we are minimizing the loss
        optimizer_discriminator.zero_grad()
        discriminator_scaler.scale(d_loss).backward()
        discriminator_scaler.step(optimizer_discriminator)
        discriminator_scaler.update()

        with torch.cuda.amp.autocast():
            d_fake = discriminator(x, y_fake)
            g_fake_loss = bce_loss(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(y, y_fake)
            g_loss = g_fake_loss + (l1 * lamda)
            total_g_loss += g_loss.item() * batch_size

        optimizer_generator.zero_grad()
        generator_scaler.scale(g_loss).backward()
        generator_scaler.step(optimizer_generator)
        generator_scaler.update()

    total_d_loss /= train_size
    total_g_loss /= train_size

    val_d_loss, val_g_loss = evaluate(discriminator, generator, val_data_loader)

    losses.append(
        [
            epoch + 1,
            total_d_loss.item(),
            val_d_loss.item(),
            total_g_loss.item(),
            val_g_loss.item(),
           
        ]
    )

    print(
        "Train D loss = {:.4f} Val D loss = {:.4f} Train G loss = {:.4f} Val G loss = {:.4f} ".format(
            total_d_loss,
            val_d_loss,
            total_g_loss,
            val_g_loss,
        )
    )

    if epoch % save_every_n_epochs == 0 or epoch == epochs - 1:
        print("=> Saving checkpoint")
        save_checkpoint(
            generator, optimizer_generator, epoch + 1, root_path, "generator"
        )
        save_checkpoint(
            discriminator,
            optimizer_discriminator,
            epoch + 1,
            root_path,
            "discriminator",
        )
        save_some_examples(generator, val_data_loader, epoch + 1, root_path, device)
        with open(root_path + "/all_log.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                "Epoch,D Train Loss,D Validation Loss,G Train Loss,G Validation Loss".split(
                    ","
                )
            )
            csvwriter.writerows(losses)
    
    loss_cols = list(zip(*losses))
    plot_curves(
        root_path,
        loss_cols[0],
        loss_cols[3],
        ylabel="Generator Loss",
    )
    plot_curves(
        root_path,
        loss_cols[0],
        loss_cols[1],
        ylabel="Discriminator Loss",
    )
    
    discriminator.train()
    generator.train()
    



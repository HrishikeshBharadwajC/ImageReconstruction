from tabnanny import verbose
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import GenerateData

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
from params import hyperparameters_CycleGAN
from utils import *
import warnings
from scores import *

warnings.filterwarnings("ignore")



epochs = hyperparameters_CycleGAN["epochs"]
lambda_x = hyperparameters_CycleGAN["lambda_x"]
lambda_y = hyperparameters_CycleGAN["lambda_y"]
batch_size = hyperparameters_CycleGAN["batch_size"]
num_workers = hyperparameters_CycleGAN["num_workers"]
learning_rate_discriminator = hyperparameters_CycleGAN["learning_rate_discriminator"]
learning_rate_generator = hyperparameters_CycleGAN["learning_rate_generator"]
generator_lr_step_size = hyperparameters_CycleGAN["generator_lr_step_size"]
discriminator_lr_step_size = hyperparameters_CycleGAN["discriminator_lr_step_size"]
epochs_initial_lr = hyperparameters_CycleGAN["epochs_initial_lr"]
epochs_decay = epochs - epochs_initial_lr
save_every_n_epochs = hyperparameters_CycleGAN["save_every_n_epochs"]
root_path = hyperparameters_CycleGAN["dataset_path"]
checkpoint_epoch = hyperparameters_CycleGAN["load_model_from_epoch"]
dataset_direction = hyperparameters_CycleGAN["dataset_direction"]
train_dataset = GenerateData(
    root_path + "/train", True if dataset_direction == "reverse" else False
)
val_dataset = GenerateData(
    root_path + "/val", True if dataset_direction == "reverse" else False
)
train_data_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
val_data_loader = DataLoader(
    dataset=val_dataset, batch_size=64, num_workers=num_workers, shuffle=False
)
device = "cuda" if torch.cuda.is_available() else "cpu"

generator_G = Generator()
generator_F = Generator()
generator_G.to(device)
generator_F.to(device)

optimizer_generator = torch.optim.Adam(
    generator_G.parameters(), lr=learning_rate_generator, betas=(0.5, 0.999)
)
generator_scaler = torch.cuda.amp.GradScaler()

discriminator_Y = Discriminator()
discriminator_Y.to(device)
discriminator_X = Discriminator()
discriminator_X.to(device)
optimizer_discriminator = torch.optim.Adam(
    discriminator_Y.parameters(), lr=learning_rate_discriminator, betas=(0.5, 0.999)
)
discriminator_scaler = torch.cuda.amp.GradScaler()
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + checkpoint_epoch - epochs_initial_lr) / float(epochs_decay + 1)
    return lr_l
# discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer_discriminator,
#     step_size=discriminator_lr_step_size,
#     gamma=0.5,
#     verbose=True,
# )
# generator_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer_generator, step_size=generator_lr_step_size, gamma=0.5, verbose=True
# )
discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer_discriminator,
    lr_lambda=lambda_rule,
    verbose=True,
)
generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer_generator, lr_lambda=lambda_rule, verbose=True
)

gan_loss = nn.MSELoss()#nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss(reduction="mean")
losses = []

if checkpoint_epoch > 0:
    discriminator_Y, optimizer_discriminator = load_checkpoint(
        root_path + f"/checkpoints/{checkpoint_epoch}_discriminator_Y.pth",
        discriminator_Y,
        optimizer_discriminator,
        learning_rate_discriminator,
        device,
    )
    discriminator_X, optimizer_discriminator = load_checkpoint(
        root_path + f"/checkpoints/{checkpoint_epoch}_discriminator_X.pth",
        discriminator_X,
        optimizer_discriminator,
        learning_rate_discriminator,
        device,
    )
    generator_G, optimizer_generator = load_checkpoint(
        root_path + f"/checkpoints/{checkpoint_epoch}_generator_G.pth",
        generator_G,
        optimizer_generator,
        learning_rate_generator,
        device,
    )
    generator_F, optimizer_generator = load_checkpoint(
        root_path + f"/checkpoints/{checkpoint_epoch}_generator_F.pth",
        generator_F,
        optimizer_generator,
        learning_rate_generator,
        device,
    )
    losses = np.loadtxt(root_path + "/all_log.csv", delimiter=",", skiprows=1).tolist()

def evaluate(discriminators, generators, val_data_loader):
    val_size = len(val_data_loader.sampler)
    total_dy_val_loss = 0.0
    total_dx_val_loss = 0.0
    total_g_val_loss = 0.0
    # total_fid = 0.0

    generator_G, generator_F = generators
    generator_G.eval()
    generator_F.eval()
    discriminator_Y, discriminator_X = discriminators
    discriminator_Y.eval()
    discriminator_X.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_data_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            batch_size = x.size(0)
            # Discriminator

            y_fake = generator_G(x)
            dy_real = discriminator_Y(x, y)
            dy_fake = discriminator_Y(x, y_fake.detach())
            x_fake = generator_F(y)
            dx_real = discriminator_X(y, x)
            dx_fake = discriminator_Y(y, x_fake.detach())

            dy_real_loss = gan_loss(dy_real, torch.ones_like(dy_real))
            dy_fake_loss = gan_loss(dy_fake, torch.zeros_like(dy_fake))
            dy_loss = (dy_real_loss + dy_fake_loss) / 2
            total_dy_val_loss += dy_loss.item() * batch_size

            dx_real_loss = gan_loss(dx_real, torch.ones_like(dx_real))
            dx_fake_loss = gan_loss(dx_fake, torch.zeros_like(dx_fake))
            dx_loss = (dx_real_loss + dx_fake_loss) / 2
            total_dx_val_loss += dx_loss.item() * batch_size

            dy_fake = discriminator_Y(x, y_fake)
            dx_fake = discriminator_X(y, x_fake)

            g_fake_loss = gan_loss(dy_fake, torch.ones_like(dy_fake))

            f_fake_loss = gan_loss(dx_fake, torch.ones_like(dx_fake))

            cycle_f_loss = l1_loss(x_fake, x) * lambda_x
            cycle_g_loss = l1_loss(y_fake, y) * lambda_y
            net_g_loss = cycle_f_loss + cycle_g_loss + g_fake_loss + f_fake_loss

            total_g_val_loss += net_g_loss.item() * batch_size

            # fid computation
            # Y_original = y
            # Y_generated = generator_G(x)
            # fid_score = calculate_fid_score(Y_original,Y_generated)
            # total_fid += fid_score

        batch_dy_loss = total_dy_val_loss / val_size
        batch_g_loss = total_g_val_loss / val_size
        batch_dx_loss = total_dx_val_loss / val_size
        # batch_fid_score = total_fid / val_size

        return batch_dy_loss, batch_g_loss, batch_dx_loss  # , batch_fid_score


train_size = len(train_data_loader.sampler)
# best_fid_score = np.inf
for epoch in range(checkpoint_epoch, epochs):
    total_dy_loss = 0.0
    total_dx_loss = 0.0
    total_g_loss = 0.0
    discriminator_Y.train()
    discriminator_X.train()
    generator_G.train()
    generator_F.train()
    # print("Epoch: ",epoch+1)
    for idx, (x, y) in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch+1}")):
        x = x.float().to(device)
        y = y.float().to(device)
        batch_size = x.size(0)
        # Discriminator
        optimizer_discriminator.zero_grad()
        with torch.cuda.amp.autocast():
            y_fake = generator_G(x)
            dy_real = discriminator_Y(x, y)
            dy_fake = discriminator_Y(x, y_fake.detach())
            x_fake = generator_F(y)
            dx_real = discriminator_X(y, x)
            dx_fake = discriminator_Y(y, x_fake.detach())

            dy_real_loss = gan_loss(dy_real, torch.ones_like(dy_real))
            dy_fake_loss = gan_loss(dy_fake, torch.zeros_like(dy_fake))
            dy_loss = (dy_real_loss + dy_fake_loss) / 2
            total_dy_loss += dy_loss.item() * batch_size

            dx_real_loss = gan_loss(dx_real, torch.ones_like(dx_real))
            dx_fake_loss = gan_loss(dx_fake, torch.zeros_like(dx_fake))
            dx_loss = (dx_real_loss + dx_fake_loss) / 2
            total_dx_loss += dx_loss.item() * batch_size

        discriminator_scaler.scale(dy_loss).backward()
        discriminator_scaler.scale(dx_loss).backward()
        discriminator_scaler.step(optimizer_discriminator)
        discriminator_scaler.update()

        # Generator
        optimizer_generator.zero_grad()
        with torch.cuda.amp.autocast():
            dy_fake = discriminator_Y(x, y_fake)
            dx_fake = discriminator_X(y, x_fake)

            g_fake_loss = gan_loss(dy_fake, torch.ones_like(dy_fake))

            f_fake_loss = gan_loss(dx_fake, torch.ones_like(dx_fake))

            cycle_f_loss = l1_loss(x_fake, x) * lambda_x
            cycle_g_loss = l1_loss(y_fake, y) * lambda_y
            net_g_loss = cycle_f_loss + cycle_g_loss + g_fake_loss + f_fake_loss

            total_g_loss += net_g_loss.item() * batch_size

        generator_scaler.scale(net_g_loss).backward()
        generator_scaler.step(optimizer_generator)
        generator_scaler.update()
    discriminator_scheduler.step()
    generator_scheduler.step()

    total_dy_loss /= train_size
    total_dx_loss /= train_size
    total_g_loss /= train_size
    val_dy_loss, val_g_loss, val_dx_loss = evaluate(
        (discriminator_Y, discriminator_X), (generator_G, generator_F), val_data_loader
    )
    # if val_fid_score < best_fid_score:
    #     save_best_weights(generator_G, optimizer_generator, epoch + 1, root_path, "generator_G")
    #     save_best_weights(
    #         discriminator_Y,
    #         optimizer_discriminator,
    #         epoch + 1,
    #         root_path,
    #         "discriminator_Y",
    #     )
    #     save_best_weights(
    #         generator_F, optimizer_generator, epoch + 1, root_path, "generator_F"
    #     )
    #     save_best_weights(
    #         discriminator_X,
    #         optimizer_discriminator,
    #         epoch + 1,
    #         root_path,
    #         "discriminator_X",
    #     )
    losses.append(
        [
            epoch + 1,
            total_dy_loss,
            val_dy_loss,
            total_dx_loss,
            val_dx_loss,
            total_g_loss,
            val_g_loss,
            # val_fid_score
        ]
    )
    print(
        "Train D_Y loss = {:.4f} Val D_Y loss = {:.4f} Train D_X loss = {:.4f} Val D_X loss = {:.4f}\nTrain G loss = {:.4f} Val G loss = {:.4f} ".format(
            total_dy_loss,
            val_dy_loss,
            total_dx_loss,
            val_dx_loss,
            total_g_loss,
            val_g_loss,
            # val_fid_score
        )
    )

    if epoch % save_every_n_epochs == 0 or epoch == epochs - 1:
        print("=> Saving checkpoint")
        save_checkpoint(
            generator_G, optimizer_generator, epoch + 1, root_path, "generator_G"
        )
        save_checkpoint(
            discriminator_Y,
            optimizer_discriminator,
            epoch + 1,
            root_path,
            "discriminator_Y",
        )
        save_checkpoint(
            generator_F, optimizer_generator, epoch + 1, root_path, "generator_F"
        )
        save_checkpoint(
            discriminator_X,
            optimizer_discriminator,
            epoch + 1,
            root_path,
            "discriminator_X",
        )
        save_some_examples(generator_G, val_data_loader, epoch + 1, root_path, device)
        with open(root_path + "/all_log.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                "Epoch,D_Y Train Loss,D_Y Validation Loss,D_X Train Loss,D_X Validation Loss,G Train Loss,G Validation Loss".split(
                    ","
                )
            )
            csvwriter.writerows(losses)

    loss_cols = list(zip(*losses))
    plot_curves(
        root_path,
        loss_cols[0],
        loss_cols[1],
        # val_values=loss_cols[2],
        ylabel="Discriminator_Y Loss",
    )
    plot_curves(
        root_path,
        loss_cols[0],
        loss_cols[3],
        # val_values=loss_cols[4],
        ylabel="Discriminator_X Loss",
    )
    plot_curves(
        root_path,
        loss_cols[0],
        loss_cols[5],
        # val_values=loss_cols[6],
        ylabel="Generator Loss",
    )

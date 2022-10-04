import torch
import config
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    font_manager.findfont(
        "Times New Roman",
        fontext="ttf",
        directory=None,
        fallback_to_default=False,
        rebuild_if_missing=False,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})
except:
    try:
        font_manager.findfont(
            "Nimbus Roman",
            fontext="ttf",
            directory=None,
            fallback_to_default=False,
            rebuild_if_missing=False,
        )
        plt.rcParams.update({"font.family": "Nimbus Roman"})
    except:
        plt.rcParams.update({"font.family": "Serif"})
plt.rcParams.update({"font.size": 14})

def scale_down(x):
    return ((x / 255) * 2) - 1


def scale_up(x):
    return ((x + 1) / 2) * 255


def plot_curves(folder, x_axis, train_values, val_values=None, xlabel="Epochs", ylabel="Loss", title=None):
    plt.figure()
    plt.plot(x_axis,train_values, label="Train")
    if val_values is not None:
        plt.plot(x_axis, val_values, label="Val")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=14)
    plt.savefig(os.path.join(folder,"results", f"{ylabel}_plot.png"), dpi=200, bbox_inches = "tight")

def plot_fids(root_path, generator_ckp_list,fids):
    plt.figure()
    plt.plot(generator_ckp_list,fids)
    plt.xlabel("Epochs")
    plt.ylabel("FID score")
    plt.savefig(root_path+'/results/fid_score.png',dpi=200, bbox_inches = "tight")
    plt.show()

def save_some_examples(generator, val_loader, epoch, folder, device):
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    generator.eval()
    with torch.no_grad():
        y_fake = generator(x)
        if not os.path.exists(folder + "/results"):
            os.mkdir(folder + "/results")
        if epoch==1:
            save_image(x, folder + "/results/input.png")
            save_image(y, folder + "/results/label.png")
            
        save_image(y_fake, folder + f"/results/y_gen_{epoch}.png")
    generator.train()


def save_checkpoint(model, optimizer, epoch, folder, model_type):

    if not os.path.exists(folder + "/checkpoints"):
        os.mkdir(folder + "/checkpoints")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, folder + f"/checkpoints/{epoch}_{model_type}.pth")


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return model, optimizer



def save_best_weights(model, optimizer, epoch, folder, model_type):
    if not os.path.exists(folder + "/checkpoints"):
        os.mkdir(folder + "/checkpoints")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    best_weights_path = folder + f"/checkpoints/best_{model_type}_weights.pth"
    torch.save(checkpoint, best_weights_path)
    print("="*100)
    print("="*100)
    print("Best Weights for the {} saved at \n {}".format(model_type,best_weights_path))
    print("="*100)
    print("="*100)
         

def write_log(root_dir, file_name, log_str):
    path = os.path.join(root_dir, file_name)
    with open(path, 'a') as f:
        f.write(log_str + '\n')

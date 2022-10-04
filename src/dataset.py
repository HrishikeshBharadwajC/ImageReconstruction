import imp
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from utils import scale_down


class GenerateData(Dataset):
    def __init__(self, path, direction_reverse=False):
        super().__init__()
        self.path = path
        self.files = os.listdir(self.path)
        self.convert_pil_to_tensor = PILToTensor()
        self.direction_reverse = direction_reverse

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = os.path.join(self.path, img_file)
        im = Image.open(img_path)
        width, height = im.size
        input_image = im.crop((0, 0, width / 2, width / 2))
        target_image = im.crop((width / 2, 0, width, height))
        if self.direction_reverse:
            input_image, target_image = target_image, input_image
        input_image = self.convert_pil_to_tensor(input_image.resize((256, 256)))
        target_image = self.convert_pil_to_tensor(target_image.resize((256, 256)))
        input_image = scale_down(input_image)
        target_image = scale_down(target_image)
        return input_image, target_image

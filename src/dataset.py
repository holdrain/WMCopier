import glob
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
from utils.helpers import set_seeds


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None, num=None ,random_sample=False, seed=2025):
        # self.data_dir = path_adapter(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = []

        if os.path.isdir(self.data_dir):
            if any(
                os.path.isdir(os.path.join(self.data_dir, d))
                for d in os.listdir(self.data_dir)
            ):
                for label, class_dir in enumerate(os.listdir(self.data_dir)):
                    class_dir_path = os.path.join(self.data_dir, class_dir)
                    if os.path.isdir(class_dir_path):
                        image_extensions = ["*.png", "*.JPEG","*.jpeg", "*.jpg", "*.webp"]
                        for ext in image_extensions:
                            self.filenames.extend(
                                glob.glob(os.path.join(class_dir_path, ext))
                            )

            else:
                image_extensions = ["*.png", "*.JPEG","*.jpeg", "*.jpg", "*.webp"]
                for ext in image_extensions:
                    self.filenames.extend(glob.glob(os.path.join(self.data_dir, ext)))

        set_seeds(seed)
        self.filenames = sorted(self.filenames)

        if num is not None:
            if random_sample:
                # print(len(self.filenames))
                self.filenames = random.sample(self.filenames,num)
            else:
                self.filenames = self.filenames[:num]
        

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.filenames)


def get_Diffusion_dl(config):
    """Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ds = CustomImageFolder(config.data_path, transform=transform, num=config.wdata_num)
    print(f"num of wm images:{ds.__len__()}")
    tdl = DataLoader(ds, batch_size=config.train_batchsize, shuffle=False)
    return tdl



import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None, num=None):

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

        self.filenames = sorted(self.filenames)

        if num is not None:
            self.filenames = self.filenames[:num]
        

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.filenames)


import os

import numpy as np
from PIL import Image
import torch.utils.data as data

from . import preprocess


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class InferenceDataset(data.Dataset):
    def __init__(self, data_folder):
        left_fold = 'imgL/'
        right_fold = 'imgR/'

        val = [x for x in os.listdir(os.path.join(data_folder, left_fold)) if is_image_file(x)]
        self.left_imgs = sorted([os.path.join(data_folder, left_fold, img) for img in val])
        self.rihgt_imgs = sorted([os.path.join(data_folder, right_fold, img) for img in val])
    
    def __getitem__(self, index):
        left  = self.left_imgs[index]
        right = self.rihgt_imgs[index]

        left_img = Image.open(left).convert("RGB")
        right_img = Image.open(right).convert("RGB")

        processed = preprocess.get_transform(augment=False)  
        left_img = processed(left_img)
        right_img = processed(right_img)

        return left_img, right_img

    def __len__(self):
        return len(self.left_imgs)

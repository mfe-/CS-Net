from __future__ import print_function, division
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
from utils.misc import ReScaleSize
import random
import warnings
import numpy as np
import scipy.misc as misc

warnings.filterwarnings('ignore')


def load_dataset(root_dir, train=True):
    images = []
    groundtruth = []
    # List all *_input.tif files and sort for reproducibility
    input_files = sorted(glob.glob(os.path.join(root_dir, '*_input.tif')))
    pairs = []
    for img_path in input_files:
        base = os.path.basename(img_path)
        subject_id = base.replace('_input.tif', '')
        gt_path = os.path.join(root_dir, f'{subject_id}_trace_mask.gif')
        if os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
        else:
            print(f'Warning: groundtruth not found for {img_path}, expected {gt_path}')

    # Split dataset: first half for training, second half for testing/validation
    n = len(pairs)
    split = n // 2
    if train:
        selected = pairs[:split]
    else:
        selected = pairs[split:]
    images = [img for img, gt in selected]
    groundtruth = [gt for img, gt in selected]
    return images, groundtruth


class Data(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 rotate=40,
                 flip=True,
                 random_crop=True,
                 scale1=512,
                 rgb=False):

        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        self.rgb = rgb
        self.images, self.groundtruth = load_dataset(self.root_dir, self.train)

    def __len__(self):
        return len(self.images)

    def RandomCrop(self, image, label, crop_size):
        crop_width, crop_height = crop_size
        w, h = image.size
        left = random.randint(0, w - crop_width)
        top = random.randint(0, h - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        new_image = image.crop((left, top, right, bottom))
        new_label = label.crop((left, top, right, bottom))
        return new_image, new_label

    def RandomEnhance(self, image):
        value = random.uniform(-2, 2)
        random_seed = random.randint(1, 4)
        if random_seed == 1:
            img_enhanceed = ImageEnhance.Brightness(image)
        elif random_seed == 2:
            img_enhanceed = ImageEnhance.Color(image)
        elif random_seed == 3:
            img_enhanceed = ImageEnhance.Contrast(image)
        else:
            img_enhanceed = ImageEnhance.Sharpness(image)
        image = img_enhanceed.enhance(value)
        return image

    def rescale(self, img, re_size):
        w, h = img.size
        min_len = min(w, h)
        new_w, new_h = min_len, min_len
        scale_w = (w - new_w) // 2
        scale_h = (h - new_h) // 2
        box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
        img = img.crop(box)
        img = img.resize((re_size, re_size))
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.groundtruth[idx]
        image = Image.open(img_path)
        label = Image.open(gt_path)

        # Convert to RGB or grayscale as needed
        if self.rgb:
            image = image.convert('RGB')
            label = label.convert('L')
        else:
            image = image.convert('L')
            label = label.convert('L')

        image = self.rescale(image, self.resize)
        label = self.rescale(label, self.resize)

        if self.train:
            # augumentation
            angel = random.randint(-self.rotate, self.rotate)
            image = image.rotate(angel)
            label = label.rotate(angel)

            if random.random() > 0.5:
                image = self.RandomEnhance(image)

            image, label = self.RandomCrop(image, label, crop_size=[self.resize, self.resize])

            # flip
            if self.flip and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        else:
            img_size = image.size
            if img_size[0] != self.resize:
                image = image.resize((self.resize, self.resize))
                label = label.resize((self.resize, self.resize))

        image = self.transform(image)
        label = self.transform(label)

        return image, label

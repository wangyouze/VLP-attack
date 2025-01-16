import json
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import dataset.image_augment as image_augment
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class pair_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            for j, caption in enumerate(ann['caption']):
                self.image.append(ann['image'])
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def tensor_to_PIL(self, tensor):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        aug_image_1 = image_augment.run(image)
        aug_image_2 = image_augment.run(image)
        aug_image_3 = image_augment.run(image)
        aug_image_4 = image_augment.run(image)
        aug_image_5 = image_augment.run(image)
        aug_image_6 = image_augment.run(image)
        aug_image_7 = image_augment.run(image)
        image = self.transform(image)
        aug_image_1 = self.transform(aug_image_1)
        aug_image_2 = self.transform(aug_image_2)
        aug_image_3 = self.transform(aug_image_3)
        aug_image_4 = self.transform(aug_image_4)
        aug_image_5 = self.transform(aug_image_5)
        aug_image_6 = self.transform(aug_image_6)
        aug_image_7 = self.transform(aug_image_7)

        aug_image = torch.cat([aug_image_1.unsqueeze(0), aug_image_2.unsqueeze(0), aug_image_3.unsqueeze(0),
                               aug_image_4.unsqueeze(0), aug_image_5.unsqueeze(0), aug_image_6.unsqueeze(0),
                               aug_image_7.unsqueeze(0)], dim=0)
        text = self.text[index]

        return image, aug_image, text, index, self.image[index]

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        image = Image.open(ann['image']).convert('RGB')
        image = self.transform(image)

        return image, caption

    @property
    def text(self):
        t = []
        for ann in self.ann:
            t += ann['caption']
        return t
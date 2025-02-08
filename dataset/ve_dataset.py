import json
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import sys
sys.path.append('/data/home/wangyouze/MultimodalAttack/VLP-attack-p/')
from VE.dataset.utils import pre_caption
import VE.dataset.image_augment as image_augment
from VE.dataset.gaussian_blur import GaussianBlur
import random

class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment': 2, 'neutral': 1, 'contradiction': 0}
        self.text = []
        self.image = []
        self.ss = []
        for i in range(len(self.ann)):
            sentence = pre_caption(self.ann[i]['sentence'], self.max_words)
            self.text.append(sentence)
            if self.ann[i]['image'] not in self.image:
                self.image.append(self.ann[i]['image'])
                self.ss.append(self.ann[i]['sentence'])

        print('the num of images:', len(self.image))

        # with open("/home/wenbo/wyz_work/Co-attack/results/ABLEF/predicted_right_ids_0408.txt", 'r',
        #           encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip().split(' ')
        #         targets_ids = [int(x) for x in line]

        self.positive_ann = []
        for j, ann in enumerate(self.ann):
            # if j not in targets_ids:
            #     continue
            if self.labels[ann['label']] == 2:
                self.positive_ann.append(ann)
        self.gaussian_blur = GaussianBlur(kernel_size=16)

    def __len__(self):
        return len(self.positive_ann)

    def __getitem__(self, index):

        ann = self.positive_ann[index]

        image_path = os.path.join(self.image_root, '%s.jpg' % ann['image'])
        image = Image.open(image_path).convert('RGB')
        # aug_img_1 = self.gaussian_blur(image)
        # aug_img_2 = self.gaussian_blur(image)

        name_list = []
        aug_img, name_list = image_augment.run(image, name_list)
        aug_img_2, name_list = image_augment.run(image, name_list)
        aug_img_3, name_list = image_augment.run(image, name_list)
        aug_img_4, name_list = image_augment.run(image, name_list)
        aug_img_5, name_list = image_augment.run(image, name_list)
        aug_img_6, name_list = image_augment.run(image, name_list)
        aug_img_7, name_list = image_augment.run(image, name_list)
        image = self.transform(image)
        aug_img = self.transform(aug_img)
        aug_img_2 = self.transform(aug_img_2)
        aug_img_3 = self.transform(aug_img_3)
        aug_img_4 = self.transform(aug_img_4)
        aug_img_5 = self.transform(aug_img_5)
        aug_img_6 = self.transform(aug_img_6)
        aug_img_7 = self.transform(aug_img_7)

        sentence = pre_caption(ann['sentence'], self.max_words)
        aug_img = torch.cat(
            [aug_img.unsqueeze(0), aug_img_2.unsqueeze(0), aug_img_3.unsqueeze(0), aug_img_4.unsqueeze(0),
             aug_img_5.unsqueeze(0), aug_img_6.unsqueeze(0), aug_img_7.unsqueeze(0)], dim=0)

        neg_img_id = random.randint(0, len(self.image) - 1)
        neg_img_path = self.image[neg_img_id]
        neg_img_path = os.path.join(self.image_root, '%s.jpg' % neg_img_path)

        neg_img = Image.open(neg_img_path).convert('RGB')
        neg_img = self.transform(neg_img)

        neg_sentence = self.ss[neg_img_id]

        return image, aug_img, neg_img, sentence, neg_sentence, self.labels[ann['label']], image_path

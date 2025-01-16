import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        self.text = []
        for i in range(len(self.ann)):
            sentence = pre_caption(self.ann[i]['sentence'], self.max_words)
            self.text.append(sentence)

        # with open("/home/wenbo/wyz_work/Co-attack/results/ABLEF/predicted_right_ids_0408.txt", 'r',
        #           encoding='utf-8') as f:
        # with open("/home/wenbo/wyz_work/Co-attack/results/TCL/TCL_predicted_right_ids_0408.txt", 'r',
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
        
    def __len__(self):
        return len(self.positive_ann)
    

    def __getitem__(self, index):    
        
        ann = self.positive_ann[index]
        
        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['sentence'], self.max_words)
       

        return image, sentence, self.labels[ann['label']], index, ann['image']
    
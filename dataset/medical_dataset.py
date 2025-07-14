import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption
import os
import random

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        with open('./data/mimic_cxr/medclip_text_features.json', 'r') as f:
            clip_features = [torch.tensor(json.loads(i.strip())) for i in f]
        self.clip_features = torch.stack(clip_features,dim=0)
       
        with open(os.path.join(image_root, 'labels_indices_train.json'), 'r') as f:
            indices = [torch.tensor(json.loads(i.strip())) for i in f]
        self.indices = torch.stack(indices,dim=0)
        
        hash_dist_path = os.path.join(image_root, 'hash_distance.json')
        self.hash_dist = json.load(open(hash_dist_path,'r'))
        with open(os.path.join(image_root, 'image_region_score_train.json'), 'r') as f:
            r_i_score = [torch.tensor(json.loads(i.strip()),dtype=torch.float32) for i in f]
        self.r_i_score = torch.stack(r_i_score,dim=0) # 270790,75
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):        
        ann = self.ann[index] 
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, 'images', image_path[0])).convert('RGB')
        image = self.transform(image)
        
        cls_labels = ann['labels']
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt)+' '
        caption = prompt + my_pre_caption(ann['report'], self.max_words)
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        clip_indices = self.indices[index]
        clip_memory = self.clip_features[clip_indices]

        region_txt = np.load(os.path.join(self.image_root, 'region_txt_embeddings', image_path[0]).replace('.jpg', '.npy'))  
        region_txt = torch.from_numpy(region_txt).to(dtype=torch.float32) # 75,768
        global_txt_path = self.ann[self.hash_dist[index][0])]['image_path'][0].replace('.jpg', '.npy')
        
        global_txt = np.load(os.path.join(self.image_root, 'medclip_txt_embeddings', global_txt_path)) # max_seq_num,768
        global_txt = torch.from_numpy(global_txt).to(dtype=torch.float32)
        local_image = self.r_i_score[index]
        return image, caption, cls_labels, clip_memory, global_txt, region_txt, local_image
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, split='val', dataset='mimic_cxr', args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else: # IU
            self.ann = self.annotation
     
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        with open('./data/mimic_cxr/medclip_text_features.json', 'r') as f:
            clip_features = [torch.tensor(json.loads(i.strip())) for i in f]
        self.clip_features = torch.stack(clip_features,dim=0)
   
        with open(os.path.join(image_root, f'labels_indices_{split}.json'), 'r') as f:
            indices = [torch.tensor(json.loads(i.strip())) for i in f]
        self.indices = torch.stack(indices,dim=0) 
        
        with open(os.path.join(image_root, f'image_region_score_{split}.json'), 'r') as f:
            r_i_score = [torch.tensor(json.loads(i.strip()),dtype=torch.float32) for i in f]
        self.r_i_score = torch.stack(r_i_score,dim=0) # 270790,75
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, 'images', image_path[0])).convert('RGB')
        image = self.transform(image)

        caption = my_pre_caption(ann['report'], self.max_words)
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels))
        clip_indices = self.indices[index]
        clip_memory = self.clip_features[clip_indices]
        ##
        if self.dataset == 'mimic_cxr':
            region_txt = np.load(os.path.join(self.image_root, 'region_txt_embeddings', image_path[0]).replace('.jpg', '.npy'))      
        else:
            region_txt = np.load(os.path.join(self.image_root, 'region_txt_embeddings', image_path[0]).replace('.png', '.npy'))   
        region_txt = torch.from_numpy(region_txt).to(dtype=torch.float32) # 75,768
        region_image = self.r_i_score[index]
        return image, caption, cls_labels, clip_memory, region_txt, region_image

        

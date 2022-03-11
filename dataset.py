from os import listdir
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import random
import pickle
from torchvision import transforms
import re

class PETDataset(Dataset):

    def __init__(self, 
                dir_path, names, 
                divided = False, 
                ):
        """
        Args:
            dir_path (string): Path to the main directory
        """
        self.dir_path = dir_path
        self.divided = divided

        self.names = names
        

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.dir_path + '/' + self.names[idx] + '.npy'
            
        images = torch.from_numpy(np.load(img_name)[:, 1, ...])
        
        trans = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(156),
                          transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)])

        images1 = [trans(images[i].unsqueeze(0)/255.) for i in range(images.shape[0])]

        images2 = np.stack(images1)
        
        if round(images2.shape[0]/320, 0)  == 2.:
            images2 = np.array([images2[i, ...] for i in range(0, images2.shape[3], 1)])
        images3 = torch.Tensor(images2.transpose((1, 2, 3, 0)))
            
        text_path = self.dir_path + '/' + self.names[idx] + '.txt.txt'
        with open(text_path) as f:
            text = f.read().rstrip().replace('\n', ' ')
        text = re.sub(r'[^\w\s]', ' ', text).replace('  ', ' ')
        
        titles = ['Пол:', '{}.*?{}'.format('Область головы', 'шеи:'),
        'Органы грудной клетки:', 'Органы брюшной полости:', 'Органы малого таза:',
        'Костная система:', 'Диагноз по МКБ-10:']
        last_title = 'Выполнение диагностической услуги:'
        
        if self.divided:
            divided_text = []
            for i in range(len(titles)):
                if i < len(titles)-1:
                    pat = re.compile('{}(.*){}'.format(titles[i], titles[i+1]))
                else:
                    pat = re.compile('{}(.*){}'.format(titles[i], last_title))
                sent = pat.findall(text)
                divided_text.append(''.join(sent).strip())

            sample = {'image': images3, 'text': divided_text}
        else:
            sample = {'image': images3, 'text': text}

        return sample


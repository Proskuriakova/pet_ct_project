import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer

from model import ModelPET
import torch.nn.functional as F
from loss import NTXentLoss

from dataset import PETDataset
from utils import collate_fn, Emb_Save
from os import listdir
from collections import OrderedDict

import logging as log
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoModel

import warnings
warnings.filterwarnings("ignore")


class PET_Model(pl.LightningModule):
    
    class DataModule(pl.LightningDataModule):
        def __init__(self, model_instance):
            super().__init__()
            self.save_hyperparameters(model_instance.hparams)
            
            self.names = []
            for dir_content in listdir(self.hparams.path_to_data):
                if dir_content.split('.')[-1] == 'npy':
                    self.names.append(dir_content.split('.')[0])

            num_train = len(self.names)
            print("LEN", num_train)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.hparams.valid_size * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]
            
            self.train_names = [self.names[i] for i in train_idx]
            self.valid_names = [self.names[i] for i in valid_idx]

        def train_dataloader(self) -> DataLoader:
            
            self.train_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.names,
                                            divided = self.hparams.divided_text, 
                                            augmentations = self.hparams.augmentations)
 
            return DataLoader(
                dataset = self.train_dataset,
                sampler = DistributedSampler(self.train_dataset),
                batch_size = self.hparams.batch_size, num_workers = self.hparams.loader_workers,
                collate_fn = collate_fn
            )



        def predict_dataloader(self) -> DataLoader:
            
            self.test_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.names,
                                            divided = self.hparams.divided_text,
                                              augmentations = self.hparams.augmentations)
             
            return DataLoader(
                dataset=self.test_dataset, sampler = RandomSampler(self.test_dataset),
                batch_size = self.hparams.batch_size, num_workers = self.hparams.loader_workers,
                collate_fn = collate_fn
            )

    def __init__(self, hparams: Namespace) -> None:
        super(PET_Model, self).__init__()
        # save_hyperparameters https://discuss.pytorch.org/t/pytorch-lightning-module-cant-set-attribute-error/121125/5
        self.save_hyperparameters(hparams)
        print(self.hparams)
 
        self.test_texts_embeds = []
        self.test_images_embeds = []
        
        self.train_texts_embeds = []
        #self.save_emdeds = Emb_Save()
        
        self.batch_size = self.hparams.batch_size
        self.out_dim = self.hparams.out_dim

        self.data = self.DataModule(self)

        self.__build_model()

        self.__build_loss()


    def __build_model(self) -> None:
        self.model_pet = ModelPET(self.hparams.image_encoder_model, self.hparams.text_encoder_model,
                             self.hparams.out_dim, self.hparams.bucket_size, 
                             self.hparams.freeze_layers, self.hparams.divided_text)

        
    def __build_loss(self):
        #data = torch.Tensor(2)
        #self._loss = nn.MSELoss()
        self._loss = NTXentLoss(self.hparams.batch_size, self.hparams.temperature,
                               self.hparams.use_cosine_similarity, self.hparams.alpha_weight)
     
    def forward(self, xis, xls):
        self.model_pet(xis, xls)
        

    def loss(self, text_embed, image_embed, batch = []) -> torch.tensor:

        return self._loss(text_embed, image_embed)

    def predict_step(self, batch: tuple, batch_nb: int) -> list:
        xls = batch['texts']
        xis = batch['images']
        name = batch['names']

        zis, zls = self.model_pet(xis, xls)
        
        return [zls.cpu().numpy(), zis.cpu().numpy(), name]
        # self.test_texts_embeds.extend(zls.cpu().numpy())
        # self.test_images_embeds.extend(zis.cpu().numpy())
        # self.predict_names.extend(name)
        
        
        
    
    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        #шаг обучения, на входе батч и номер батча
        #возвращает loss и инфу для логгера pl

        xls = batch['texts']
        xis = batch['images'] 
        
        zis, zls = self.model_pet(xis, xls, mode = 'train')
        #print(zls)
        self.train_texts_embeds.append(zls.detach().cpu().numpy())
        loss_val = self._loss(zis, zls)

        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)      

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    
    def test_step(self, batch: list, batch_nb: int, *args, **kwargs) -> dict:
        with torch.no_grad():
            self.model_pet.eval()
          
            xls = batch['texts']
            xis = batch['images']  
            names = batch['names']

            zis, zls = self.model_pet(xis, xls, mode = 'train')
            test_loss = self._loss(zis, zls)
        
            #self.save_emdeds.update(zls, zis, names)
        
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)     
 

        
        tqdm_dict = {"test_loss": test_loss}
        
        output = OrderedDict({"test_loss": test_loss, 
                             "progress_bar": tqdm_dict, "log": tqdm_dict})

        return output
    
#     def test_epoch_end(self):
        
#         self.save_emdeds.compute(self.hparams.out_name)
        
#         print('ALL', len(self.test_texts_embeds))
#         print('zero element', len(self.test_texts_embeds[0]))
#         texts_embeds = np.array(self.test_texts_embeds)
#         with open('texts_embeddings_bs3.npy', 'wb') as f:
#             np.save(f, texts_embeds)
#         images_embeds = np.array(self.test_images_embeds)
#         with open('images_embeddings_bs3.npy', 'wb') as f:
#             np.save(f, images_embeds)
                 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model_pet.parameters(), lr = self.hparams['learning_rate'])
        return [optimizer], []
        #default = "DeepPavlov/rubert-base-cased-sentence",

     
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        
        parser.add_argument(
            "--path_to_data",
            default = "/data/burenko/datasets/pet-ct",
            type = str,
            help = "Path to the data",
        )
        parser.add_argument(
            "--text_encoder_model",
            default = 'sberbank-ai/ruRoberta-large',
            type = str,
            help = "Text encoder",
        )
        parser.add_argument(
            "--image_encoder_model",
            default = "resnet18_3D",
            type = str,
            help = "Image encoder",
        )        
        parser.add_argument(
            "--temperature",
            default = 1e-03,
            type = float,
            help = "Temperature for loss calculation",
        )
        parser.add_argument(
            "--learning_rate",
            default = 1e-05,
            type = float,
            help = "learning rate",
        )
        parser.add_argument(
            "--valid_size",
            default = 0.2,
            type = float,
            help = "Size of validation sample",
        )        
        parser.add_argument(
            "--use_cosine_similarity",
            default = True,
            type = bool,
            help = "Using cosine similarity in loss or not - bool, default True",
        )
        parser.add_argument(
            "--alpha_weight",
            default = 0.5,
            type = int,
            help = "Loss parameter, default = 0.75",
        )

        parser.add_argument(
            "--bucket_size",
            default = 32,
            type = np.int64,
            help = "Count of images processing per time, default - 32,"\
            "if None - all images per patient processing together (need big GPU)",
        )
        parser.add_argument(
            "--out_dim",
            default = 300,
            type = np.int64,
            help = "Size of output embeddings, default - 300",
        )
        parser.add_argument(
            "--freeze_layers",
            default = [0, 1, 2, 3],
            type = list,
            help = "",
        )        
        parser.add_argument(
            "--loader_workers",
            default = 1,
            type = int,
            help = "Count of workers",
        )
        parser.add_argument(
            "--divided_text",
            default = True,
            type = bool,
            help = "Divide text into logical parts or not, bool, default - True",
        )        

        parser.add_argument(
            "--out_name",
            default = 'tmp',
            type = str,
            help = "Name for output files",
        )      
        parser.add_argument(
            "--augmentations",
            default = False,
            type = bool,
            help = "Name for output files",
        )     
        return parser
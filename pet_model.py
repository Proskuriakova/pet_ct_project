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
from utils import collate_fn
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
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.hparams.valid_size * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]
            
            self.train_names = [self.names[i] for i in train_idx]
            self.valid_names = [self.names[i] for i in valid_idx]
            #print(len(self.train_names))
            #print(len(self.valid_names))
            #print(self.names[:2])

        def train_dataloader(self) -> DataLoader:
            
            self.train_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.names,
                                            divided = self.hparams.divided_text)
 
            return DataLoader(
                dataset = self.train_dataset,
                sampler = DistributedSampler(self.train_dataset),
                batch_size = self.hparams.batch_size, num_workers = self.hparams.loader_workers,
                collate_fn = collate_fn
            )

#         def val_dataloader(self) -> DataLoader:
        
#             self.val_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.valid_names,
#                                             divided = self.hparams.divided_text)
             
#             return DataLoader(
#                 dataset = self.val_dataset, sampler = DistributedSampler(self.val_dataset),
#                 batch_size=self.hparams.batch_size, num_workers = self.hparams.loader_workers,
#                 collate_fn = collate_fn
#             )

        def test_dataloader(self) -> DataLoader:
            
            self.test_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.names,
                                            divided = self.hparams.divided_text)
             
            return DataLoader(
                dataset=self.test_dataset, sampler = DistributedSampler(self.test_dataset),
                batch_size = self.hparams.batch_size, num_workers = self.hparams.loader_workers,
                collate_fn = collate_fn
            )

    def __init__(self, hparams: Namespace) -> None:
        super(PET_Model, self).__init__()
        # save_hyperparameters https://discuss.pytorch.org/t/pytorch-lightning-module-cant-set-attribute-error/121125/5
        self.save_hyperparameters(hparams)
        print(self.hparams)
 
        
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

 
    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        #шаг обучения, на входе батч и номер батча
        #возвращает loss и инфу для логгера pl

        xls = batch['texts']
        xis = batch['images'] 
        

        zis, zls = self.model_pet(xis, xls)
        loss_val = self._loss(zis, zls)

        
        self.log('train_loss', loss_val, on_epoch=True, logger=True)      

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

#     def validation_step(self, batch: list, batch_nb: int, *args, **kwargs) -> dict:

#         xls = batch['texts']
#         xis = batch['images']        

#         zis, zls = self.model_pet(xis, xls)
#         loss_val = self._loss(zis, zls)

        
#         self.log('val_loss', loss_val, on_epoch=True, logger=True)      
 

        
#         tqdm_dict = {"val_loss": loss_val}
        
#         output = OrderedDict({"val_loss": loss_val, 
#                              "progress_bar": tqdm_dict, "log": tqdm_dict})

#         return output

    
    def test_step(self, batch: list, batch_nb: int, *args, **kwargs) -> dict:
        with torch.no_grad():
            self.model_pet.eval()
          
            xls = batch['texts']
            xis = batch['images']        

            zis, zls = self.model_pet(xis, xls)
            loss_val = self._loss(zis, zls)

        
        self.log('val_loss', loss_val, on_epoch=True, logger=True)      
 

        
        tqdm_dict = {"val_loss": loss_val}
        
        output = OrderedDict({"val_loss": loss_val, 
                             "progress_bar": tqdm_dict, "log": tqdm_dict})

        return output
        
         

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model_pet.parameters(), lr = self.hparams['learning_rate'])
        return [optimizer], []

     
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
            default = "DeepPavlov/rubert-base-cased-sentence",
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
            default = 0.01,
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
            default = 0.75,
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
            default = None,
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



        return parser
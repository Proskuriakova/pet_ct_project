import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from transformers import AutoModel, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

def pprint_snapshot():
    s = torch.cuda.memory_snapshot()
    for seg in s:
        print("%7.2f | %7.2f MB - %s" % (
            seg["active_size"] / 1000000., seg["total_size"] / 1000000., seg["segment_type"]))
        for b in seg["blocks"]:
            print("    %7.2f MB - %s" % (b["size"] / 1000000., b["state"]))

class ModelPET(nn.Module):
    def __init__(self, res_base_model, bert_base_model, out_dim, bucket_size, freeze_layers, divided):
        super(ModelPET, self).__init__()
        self.out_dim = out_dim
        self.bucket_size = bucket_size
        self.divided = divided
        self.l_dim = int(320 / self.bucket_size)
        
        #init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_base_model)
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(1024, 1024) #1024 is the size of the BERT embbedings (312 for tiny)
        self.bert_l2 = nn.Linear(1024, out_dim) #1024 is the size of the BERT embbedings

        # init Resnet
        self.resnet_dict = {"resnet18_3D": models.video.r3d_18(pretrained = False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_res_basemodel(res_base_model)
        self.res_base_model = res_base_model
        self.num_ftrs = resnet.fc.in_features
        self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        # projection MLP for ResNet Model
        self.res_l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
        self.res_l2 = nn.Linear(self.num_ftrs, self.out_dim)
        #concat images projections per patient
        self.im_l1_concat = nn.Linear(self.out_dim*self.l_dim, self.out_dim)
        self.im_l2_concat = nn.Linear(self.out_dim, self.out_dim)
        #concat text projections per patient
        self.txt_l1_concat = nn.Linear(self.out_dim*7, self.out_dim)
        self.txt_l2_concat = nn.Linear(self.out_dim, self.out_dim)
        
        
    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def image_encoder(self, xis):
                
        h = self.res_features(xis)
        x = h.squeeze()
        x = self.res_l1(x)
        x = F.relu(x)
        x = self.res_l2(x)
 
        return x
    
    
    def concat_embed_model(self, embeds, mode):
        if mode == 'img':
            zis = self.im_l1_concat(embeds)
            zis = F.relu(zis)
            zis = self.im_l2_concat(zis)
        else:
            zis = self.txt_l1_concat(embeds)
            zis = F.relu(zis)
            zis = self.txt_l2_concat(zis)
        return zis
        
        
    def text_encoder(self, inputs, mode):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        encoded_inputs = self.tokenizer(inputs, 
                                         return_tensors="pt", 
                                         padding=True,
                                         truncation=True, max_length = 512).to(next(self.bert_model.parameters()).device)
        #print('INPUT IDS', encoded_inputs['input_ids'].get_device())
        #print('MODEL ', next(self.bert_model.parameters()).device)
        outputs = self.bert_model(**encoded_inputs)
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
        sentence_embeddings.to(torch.half)    
        if mode == 'train':
            x = self.bert_l1(sentence_embeddings.to(torch.half))
        else:
            x = self.bert_l1(sentence_embeddings)
        
        #x = self.bert_l1(sentence_embeddings.to(torch.half))
        x = F.relu(x)
        out_emb = self.bert_l2(x)

        return out_emb


    def text_encode(self, text, mode):
        if self.divided:
            slice_embeds = []
            for j in range(len(text)):
                slice_embeds.append(self.text_encoder(text[j], mode).squeeze())
            x = torch.cat(slice_embeds, dim = 0)
        else:
            x = self.text_encoder(text)
        
        return x
    
    def image_encode(self, xis):
        if self.res_base_model == 'resnet18_3D':
            if self.bucket_size is not None:
                slice_imbeds = []
                for j in range(0, xis.shape[3], self.bucket_size):
                    xis_j = xis[..., j:j + self.bucket_size]
                    h_j = self.image_encoder(xis_j.unsqueeze(0))
                    slice_imbeds.append(h_j.squeeze())
                    
                if len(slice_imbeds) <= self.l_dim:
                    for i in range(self.l_dim - len(slice_imbeds)):
                        slice_imbeds.append(torch.zeros_like(torch.empty(300)).type_as(slice_imbeds[0]))
                else:
                    slice_imbeds = slice_imbeds[:self.l_dim]
                x = torch.cat(slice_imbeds, dim = 0)
                    
            else:    
                x = self.image_encoder(xis.unsqueeze(0))
        else:
            if self.bucket_size is not None:
                slice_imbeds = []
                for j in range(0, xis.shape[3], self.bucket_size):
                    xis_j = xis[..., j:j + self.bucket_size]
                    h_j = [self.image_encoder(xis_j[..., i].unsqueeze(0)).squeeze() for i in range(xis_j.shape[3])]
                    x_j = torch.cat(h_j, dim = 0)
                    x_j = self.concat_embed_model(x_j)
                    slice_imbeds.append(x_j.squeeze())

                x = torch.cat(slice_imbeds, dim = 0)
                
            else:    
                image_embeds = []
                for i in range(xis.shape[3]):
                    xis_i = xis[..., i]
                    h_i = self.image_encoder(xis_i.unsqueeze(0))
                    image_embeds.append(h_i.squeeze())
                x = torch.cat(image_embeds, dim = 0)
        return x
  
            
    def forward(self, images_batch, text_batch, mode):
        
        zis = [self.concat_embed_model(self.image_encode(images_batch[i]), mode = 'img') for i in range(len(images_batch))]
        zis_stack = torch.stack(zis, dim=0).float()
        #image_embed = self.concat_embed_model(zis_stack)
        # print('SHAPE EMBED', image_embed.shape)

        if self.divided:
            zls = [self.concat_embed_model(self.text_encode(text_batch[i], mode), mode = 'txt') for i in range(len(text_batch)) ]
        else:
            zls = [self.text_encode(text_batch[i], mode).squeeze() for i in range(len(text_batch)) ]
        zls_stack = torch.stack(zls, dim=0).float()
        #text_embed = self.concat_embed_model(zls_stack)
        
        return zis_stack.squeeze(1), zls_stack.squeeze(1)
        #, zis_stack.squeeze(1)        
        
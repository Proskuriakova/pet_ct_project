import torchmetrics

def collate_fn(data):

    texts = [data[i]['text'] for i in range(len(data))]
    images = [data[i]['image'] for i in range(len(data))]
    names = [data[i]['name'] for i in range(len(data))]
    data = {'texts': texts, 'images': images, 'names': names}
#     max_l_txt = np.max([len(texts[i]) for i in range(len(texts))])
#     max_l_img = np.max([images[i].shape[3] for i in range(len(images))])


#     for i in range(len(data)):
#         img = data[i]['image']
#         added = torch.zeros(list(img.shape[:-1]) + [max_l_img - img.shape[-1]])
#         data[i]['image'] = torch.cat([data[i]['image'],added], dim = 3)

    return data


class Emb_Save(torchmetrics.Metric):
    def __init__(self):
        self.txt_embeds = []
        self.img_embeds, self.names = [], []
        
    def update(self, texts, images, names):
        self.txt_embeds.extend(texts)
        self.img_embeds.extend(images)
        self.names.append(names)
        
    def compute(self, file_name):
        img_name = 'results/image_embeddings_' + file_name + '.npy'
        txt_name = 'results/text_embeddings_' + file_name + '.npy'
        f_name = 'results/names_' + file_name + '.txt'
        
        texts_embeds = np.array(self.txt_embeds)
        with open(img_name, 'wb') as f:
            np.save(f, self.txt_embeds)
        images_embeds = np.array(self.img_embeds)
        with open(txt_name, 'wb') as f:
            np.save(f, self.img_embeds)  
        with open(f_name, 'w') as f:
            for item in self.names:
                f.write("%s\n" % item)
    
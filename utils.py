def collate_fn(data):

    texts = [data[i]['text'] for i in range(len(data))]
    images = [data[i]['image'] for i in range(len(data))]
    data = {'texts': texts, 'images': images}
#     max_l_txt = np.max([len(texts[i]) for i in range(len(texts))])
#     max_l_img = np.max([images[i].shape[3] for i in range(len(images))])


#     for i in range(len(data)):
#         img = data[i]['image']
#         added = torch.zeros(list(img.shape[:-1]) + [max_l_img - img.shape[-1]])
#         data[i]['image'] = torch.cat([data[i]['image'],added], dim = 3)

    return data
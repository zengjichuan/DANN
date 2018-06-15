import torch.utils.data as data
import os
import json
import logging
import gensim

logger = logging.getLogger()

class GetLoader(data.Dataset):
    def __init__(self, data_fn, transform=None):
        self.data_fn = data_fn
        self.transform = transform

        j_data = json.load(open(data_fn, 'r'))
        self.data = j_data["data"]
        self.n_data = len(self.data)
        self.posts = []
        self.labels = []
        self.label_dict = {}
        for item in self.data:
            post = item["title"] + " " + item["text"]
            self.posts.append(gensim.utils.tokenize(post, lower=True))
            label = item["category"]
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict)
            self.labels.append(self.label_dict[label])

        logger.info("Loaded %s dataset, totally %d items." % (j_data["dataset"], self.n_data))


    def __getitem__(self, item):
        post, label = self.posts[item], self.labels[item]

        if self.transform is not None:
            post = self.transform(post)
            label = int(label)

        return post, label


    def __len__(self):
        return self.n_data

    def dump_label_dict(self):
        # dump label dict
        json.dump(self.label_dict, open(os.path.join(os.path.dirname(self.data_fn), "label_dict.json")))
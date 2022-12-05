"""Data provider"""

import torch
import torch.utils.data as data
import os
import nltk
import numpy as np
import codecs


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """
    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'
        self.data_split = data_split

        # Captions
        self.captions = []
        with codecs.open(loc + '%s_caps.txt' % data_split, 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_cnns.npy' % data_split)
        self.bovws = np.load(loc + '%s_bovws.npy' % data_split)
        self.labels = np.load(loc + '%s_labels.npy' % data_split)
        self.length = len(self.captions)

        print("split: %s, total images_CNN: %d, total images_BovW: %d, total captions: %d"
              % (data_split, self.images.shape[0], self.bovws.shape[0], self.length))

    def __getitem__(self, index):
        # handle the image redundancy
        image = torch.Tensor(self.images[index])
        bovw = torch.Tensor(self.bovws[index])
        lab = torch.Tensor(self.labels[index])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())

        caption = [vocab('<start>')]
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, index, lab, bovw

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, lab, bovws = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    bovws = torch.stack(bovws, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.LongTensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), max(lengths)).long()
    masks = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        masks[i, :end] = 1

    return images, targets, lengths, masks, ids, lab, bovws


def get_precomp_loader(data_path, data_split, vocab, batch_size=100, shuffle=True):
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, batch_size, True)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, batch_size, False)
    return train_loader, val_loader


def get_test_loader(data_name, vocab, batch_size, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, 'test', vocab, batch_size, False)
    return test_loader

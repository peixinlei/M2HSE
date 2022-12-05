"""Evaluation"""
from __future__ import print_function
import os
import numpy as np
import torch


def logging_func(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message)
    f.close()


def compute_sim_score(im, s):
    im_T = torch.transpose(im, 0, 1)  # (embed_size,batch_size_im)
    scores = torch.mm(s, im_T)
    return scores


def compute_sims(images, captions, shard_size=128):
    """
    Computer pairwise t2i distance with locality sharding
    """
    n_im_shard = (len(images) - 1) / shard_size + 1
    n_cap_shard = (len(captions) - 1) / shard_size + 1

    d = np.zeros((len(captions), len(images)))
    for i in range(int(n_im_shard)):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(int(n_cap_shard)):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            sim = compute_sim_score(im, s)
            d[cap_start:cap_end, im_start:im_end] = sim.data.cpu().numpy()
    return d.T


def shard_xattn_data(model, cnns, captions, caplens, opt):
    sims_0 = shard_xattn_t2i(model, cnns, captions, caplens, opt, shard_size=128)
    sims_1 = shard_xattn_i2t(model, cnns, captions, caplens, opt, shard_size=128)
    sims = sims_0 + sims_1
    return sims


def shard_xattn_t2i(model, images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.xattn_score_t2i(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d


def shard_xattn_i2t(model, images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.xattn_score_i2t(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d


def i2t_mAP(sims, targets):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    queryResult = sims
    # qurCat = np.array([targets[i] for i in range(0, len(targets), 5)])
    qurCat = targets
    resCat = targets

    queryResult = queryResult.argsort()[:, ::-1]
    AP = np.zeros((len(qurCat), 1))
    recall = np.zeros((len(qurCat), len(resCat)))
    precision = np.zeros((len(qurCat), len(resCat)))

    for i in range(len(qurCat)):
        resultList = queryResult[i]
        relCount = 0
        relAll = 0
        for k in resultList:
            if resCat[k] == qurCat[i]:
                relAll = relAll + 1

        for j in range(len(resCat)):
            if resCat[resultList[j]] == qurCat[i]:
                relCount = relCount + 1
                AP[i] = AP[i] + relCount / (j + 1)
                precision[i, j] = relCount / (j + 1)
                recall[i, j] = relCount / relAll
                if recall[i, j] > 1:
                    print('recall > 1!')
                    break
            else:
                precision[i, j] = relCount / (j + 1)
                recall[i, j] = relCount / relAll
        AP[i] = AP[i] / relCount
    mAP = np.mean(AP)
    return mAP


def t2i_mAP(sims, targets):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    queryResult = sims.T
    qurCat = targets
    # resCat = np.array([targets[i] for i in range(0, len(targets), 5)])
    resCat = targets

    queryResult = queryResult.argsort()[:, ::-1]
    AP = np.zeros((len(qurCat), 1))
    recall = np.zeros((len(qurCat), len(resCat)))
    precision = np.zeros((len(qurCat), len(resCat)))

    for i in range(len(qurCat)):
        resultList = queryResult[i]
        relCount = 0
        relAll = 0
        for k in resultList:
            if resCat[k] == qurCat[i]:
                relAll = relAll + 1

        for j in range(len(resCat)):
            if resCat[resultList[j]] == qurCat[i]:
                relCount = relCount + 1
                AP[i] = AP[i] + relCount / (j + 1)
                precision[i, j] = relCount / (j + 1)
                recall[i, j] = relCount / relAll
                if recall[i, j] > 1:
                    print('recall > 1!')
                    break
            else:
                precision[i, j] = relCount / (j + 1)
                recall[i, j] = relCount / relAll
        AP[i] = AP[i] / relCount
    mAP = np.mean(AP)
    return mAP


def recall_K(sims, targets, K_mean):
    queryResult = sims
    qurCat = targets
    resCat = targets

    queryResult = queryResult.argsort()[:, ::-1]
    recall = np.zeros((len(qurCat), len(resCat)))
    recallk = np.zeros((len(qurCat), 1))
    for i in range(len(qurCat)):
        resultList = queryResult[i]
        relCount = 0
        relAll = 0
        for k in resultList:
            if resCat[k] == qurCat[i]:
                relAll = relAll + 1

        for j in range(len(resCat)):
            if resCat[resultList[j]] == qurCat[i]:
                relCount = relCount + 1
                recall[i, j] = relCount / relAll
                if recall[i, j] > 1:
                    print('recall > 1!')
                    break
            else:
                recall[i, j] = relCount / relAll

    for i in range(len(qurCat)):
        resultList = queryResult[i]
        relAll = 0
        for k in resultList:
            if resCat[k] == qurCat[i]:
                relAll = relAll + 1
        for j in range(K_mean):
            if resCat[resultList[j]] == qurCat[i]:
                recallk[i] = 1
    Recall_K = np.mean(recallk)
    return Recall_K


def encode_global_data(model, data_loader):
    global_cnn_embs = None
    global_gru_embs = None
    global_labels = None
    max_n_word = 0
    for i, (cnns, captions, lengths, masks, ids, lab, bovws) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (cnns, captions, lengths, masks, ids, lab, bovws) in enumerate(data_loader):

        cnn_embs, gru_embs = model.forward_emb_cnn(cnns, captions, lengths)

        if global_cnn_embs is None:
            global_cnn_embs = np.zeros((len(data_loader.dataset), cnn_embs.size(1)))
            global_gru_embs = np.zeros((len(data_loader.dataset), gru_embs.size(1)))
            global_labels = np.zeros((len(data_loader.dataset)))

        # cache embeddings
        global_cnn_embs[ids] = cnn_embs.data.cpu().numpy().copy()
        global_gru_embs[ids] = gru_embs.data.cpu().numpy().copy()
        global_labels[ids] = lab

    return global_cnn_embs, global_gru_embs, global_labels


def encode_local_data(model, data_loader):
    """Encode all cnns and captions loadable by `data_loader`
    """
    # np array to keep all the embeddings
    cnn_embs = None
    cnn_label = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (cnns, captions, lengths, masks, ids, lab, bovws) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (cnns, captions, lengths, masks, ids, lab, bovws) in enumerate(data_loader):

        # compute the embeddings
        cnn_emb, cap_emb, cap_len = model.forward_emb_cnn(cnns, captions, lengths)
        if cnn_embs is None:
            if cnn_emb.dim() == 3:
                cnn_embs = np.zeros((len(data_loader.dataset), cnn_emb.size(1), cnn_emb.size(2)))
                cnn_label = np.zeros((len(data_loader.dataset)))
            else:
                cnn_embs = np.zeros((len(data_loader.dataset), cnn_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        cnn_embs[ids] = cnn_emb.data.cpu().numpy().copy()
        cnn_label[ids] = lab
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
        del cnns, captions
    return cnn_embs, cap_embs, cap_lens, cnn_label


def eval_global_mAP(model, data_loader, opt, split='dev'):
    print("-------- evaluation --------")
    model.eval()
    with torch.no_grad():
        global_cnn_embs, global_gru_embs, global_labs = encode_global_data(model, data_loader)
        print('global_Images: %d, global_Texts: %d' % (global_cnn_embs.shape[0], global_gru_embs.shape[0]))

        sim_P_G = compute_sims(global_cnn_embs, global_gru_embs, shard_size=128)
        mAP_i2t = i2t_mAP(sim_P_G, global_labs)
        mAP_t2i = t2i_mAP(sim_P_G, global_labs)
        mAP_sum = mAP_i2t + mAP_t2i

        print("===============================")
        print("i2t mAP %.4f" % mAP_i2t)
        print("t2i mAP %.4f" % mAP_t2i)
        print("mAP_sum: %.4f" % mAP_sum)
        print("===============================")

        message = "split: %s \n" \
                  "Image to text: %.4f " \
                  "Text to text: %.4f " \
                  "mAP_sum: %.4f " \
                  % (split, mAP_i2t, mAP_t2i, mAP_sum)

        log_file = os.path.join(opt.logger_name, "performance.log")
        logging_func(log_file, message)
        return mAP_sum


def eval_local_mAP(model, data_loader, opt, split='dev'):
    print("-------- evaluation --------")
    model.eval()
    with torch.no_grad():
        local_cnn_embs, local_gru_embs, local_gru_lens, local_labs = encode_local_data(model, data_loader)
        print('local_Images: %d, local_Texts: %d' % (local_cnn_embs.shape[0], local_gru_embs.shape[0]))

        sim_P = shard_xattn_data(model, local_cnn_embs, local_gru_embs, local_gru_lens, opt)
        mAP_i2t = i2t_mAP(sim_P, local_labs)
        mAP_t2i = t2i_mAP(sim_P, local_labs)
        mAP_sum = mAP_i2t + mAP_t2i

        print("===============================")
        print("i2t mAP %.4f" % mAP_i2t)
        print("t2i mAP %.4f" % mAP_t2i)
        print("mAP_sum: %.4f" % mAP_sum)
        print("===============================")

        message = "split: %s \n" \
                  "Image to text: %.4f " \
                  "Text to text: %.4f " \
                  "mAP_sum: %.4f " \
                  % (split, mAP_i2t, mAP_t2i, mAP_sum)

        log_file = os.path.join(opt.logger_name, "performance.log")
        logging_func(log_file, message)
        return mAP_sum







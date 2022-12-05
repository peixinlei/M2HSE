import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X + eps, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


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

    d = torch.zeros((len(captions), len(images))).cuda()
    for i in range(int(n_im_shard)):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(int(n_cap_shard)):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = images[im_start:im_end]
            s = captions[cap_start:cap_end]
            sim = compute_sim_score(im, s)
            d[cap_start:cap_end, im_start:im_end] = sim
    return d.T


def func_attention(query, context, opt, smooth):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)
    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)
    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class EncoderImage(nn.Module):

    def __init__(self, img_dim, embed_size):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        features = l2norm(features, dim=-1)
        return features


class EncoderText_global(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=True):
        super(EncoderText_global, self).__init__()
        self.embed_size = embed_size
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        packed = pack_padded_sequence(x_emb, lengths.data.tolist(), batch_first=True)

        # Forward propagate RNN
        out, ht = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        cap_emb = l2norm(cap_emb, dim=-1)

        features = torch.mean(cap_emb, dim=1)
        global_features = features.view(cap_emb.shape[0], self.embed_size)

        return global_features


class EncoderText_local(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=True):
        super(EncoderText_local, self).__init__()
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True,
                          bidirectional=use_bi_gru)
        # weight initial
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        packed = pack_padded_sequence(x_emb, lengths.data.tolist(), batch_first=True)

        # Forward propagate RNN
        out, ht = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb


class MMHSE_global(nn.Module):

    def __init__(self, opt):
        super(MMHSE_global, self).__init__()
        self.opt = opt
        self.global_cnn_enc = EncoderImage(opt.cnn_dim, opt.embed_size)
        self.global_bovw_enc = EncoderImage(opt.bovw_dim, opt.embed_size)
        self.global_cnn_gru_enc = EncoderText_global(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, opt.bi_gru)
        self.global_bovw_gru_enc = EncoderText_global(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, opt.bi_gru)

    def forward_emb_cnn(self, cnns, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            cnns = cnns.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
        # Forward
        global_cnn_embs = self.global_cnn_enc(cnns)
        global_gru_embs = self.global_cnn_gru_enc(captions, lengths)
        return global_cnn_embs, global_gru_embs

    def forward_emb_bovw(self, bovws, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            bovws = bovws.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
        # Forward
        global_bovw_embs = self.global_bovw_enc(bovws)
        global_gru_embs = self.global_bovw_gru_enc(captions, lengths)
        return global_bovw_embs, global_gru_embs

    def forward(self, cnns, captions, lengths, bovws):
        """One training step given images and captions.
        """
        # compute the embeddings
        global_cnn_embs, global_cnn_gru_embs = self.forward_emb_cnn(cnns, captions, lengths)
        global_bovw_embs, global_bovw_gru_embs = self.forward_emb_bovw(bovws, captions, lengths)
        scores_0 = compute_sims(global_cnn_embs, global_cnn_gru_embs, shard_size=128)
        scores_1 = compute_sims(global_bovw_embs, global_bovw_gru_embs, shard_size=128)
        scores_2 = compute_sims(global_cnn_embs, global_bovw_embs, shard_size=128)
        return scores_0, scores_1, scores_2


class MMHSE_local(nn.Module):

    def __init__(self, opt):
        super(MMHSE_local, self).__init__()
        # Build Models
        self.opt = opt
        self.local_cnn_enc = EncoderImage(opt.cnn_dim, opt.embed_size)
        self.local_bovw_enc = EncoderImage(opt.bovw_dim, opt.embed_size)
        self.local_cnn_gru_enc = EncoderText_local(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, opt.bi_gru)
        self.local_bovw_gru_enc = EncoderText_local(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, opt.bi_gru)

    def forward_emb_cnn(self, cnns, captions, lengths):
        # Set mini-batch dataset
        if torch.cuda.is_available():
            cnns = cnns.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
        # Forward
        cnn_emb = self.local_cnn_enc(cnns)
        cap_emb = self.local_cnn_gru_enc(captions, lengths)

        return cnn_emb, cap_emb, lengths

    def forward_emb_bovw(self, bovws, captions, lengths):
        # Set mini-batch dataset
        if torch.cuda.is_available():
            bovws = bovws.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
        # Forward
        bovw_emb = self.local_bovw_enc(bovws)
        cap_emb = self.local_bovw_gru_enc(captions, lengths)
        return bovw_emb, cap_emb, lengths

    def forward_score(self, img_emb, cap_emb, cap_len):
        scores_0 = self.xattn_score_t2i(img_emb, cap_emb, cap_len, self.opt)
        scores_1 = self.xattn_score_i2t(img_emb, cap_emb, cap_len, self.opt)

        scores = scores_0 + scores_1

        return scores

    def forward_score_cb(self, c, b):
        scores_0 = self.xattn_score_c2b(c, b, self.opt)
        scores_1 = self.xattn_score_b2c(c, b, self.opt)
        scores = scores_0 + scores_1

        return scores

    def forward(self, cnns, captions, lengths, bovws):
        # Compute the Embeddings
        cnn_emb, cnn_cap_emb, cnn_cap_lens = self.forward_emb_cnn(cnns, captions, lengths)
        bovw_emb, bovw_cap_emb, bovw_cap_lens = self.forward_emb_bovw(bovws, captions, lengths)
        # Compute the Cross-modal Similarity Matrices
        Similarity_P = self.forward_score(cnn_emb, cnn_cap_emb, cnn_cap_lens)
        Similarity_A1 = self.forward_score(bovw_emb, bovw_cap_emb, bovw_cap_lens)
        Similarity_A2 = self.forward_score_cb(cnn_emb, bovw_emb)

        return Similarity_P, Similarity_A1, Similarity_A2

    def xattn_score_t2i(self, images, captions_all, cap_lens, opt):
        similarities = []
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()

        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()

            row_sim = cosine_similarity(cap_i_expand.double(), weiContext.double(), dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)

            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).double()

        return similarities

    def xattn_score_i2t(self, images, captions_all, cap_lens, opt):
        similarities = []
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()

        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)

            row_sim = cosine_similarity(images, weiContext, dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).double()

        return similarities

    def xattn_score_c2b(self, cnn, bovw, opt):
        similarities = []
        n_c = cnn.size(0)
        n_b = bovw.size(0)
        cnn = cnn.float()
        bovw = bovw.float()
        for i in range(n_b):
            n_p = 9
            b_i = bovw[i, :n_p, :].unsqueeze(0).contiguous()
            b_i_expand = b_i.repeat(n_c, 1, 1)

            weiContext, attn = func_attention(cnn, b_i_expand, opt, smooth=opt.lambda_softmax)

            row_sim = cosine_similarity(cnn, weiContext, dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)
            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1).double()
        return similarities

    def xattn_score_b2c(self, cnn, bovw, opt):

        similarities = []
        n_c = cnn.size(0)
        n_b = bovw.size(0)
        cnn = cnn.float()
        bovw = bovw.float()

        for i in range(n_b):
            n_p = 9
            b_i = bovw[i, :n_p, :].unsqueeze(0).contiguous()
            b_i_expand = b_i.repeat(n_c, 1, 1)

            weiContext, attn = func_attention(b_i_expand, cnn, opt, smooth=opt.lambda_softmax)

            b_i_expand = b_i_expand.contiguous()
            weiContext = weiContext.contiguous()

            row_sim = cosine_similarity(b_i_expand.double(), weiContext.double(), dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)
            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1).double()
        return similarities


class MultiSpringBalanceLoss(nn.Module):
    def __init__(self, opt):
        super(MultiSpringBalanceLoss, self).__init__()
        self.Delta = opt.Delta
        self.gamma_1 = opt.gamma_1
        self.gamma_2 = opt.gamma_2

    def forward(self, scores, labels):
        batch_size = scores.size(0)
        scoresT = scores.t()
        targets = torch.cat(labels)
        a1 = targets.expand(batch_size, batch_size)
        a2 = targets.expand(batch_size, batch_size).t()
        mask = a1.eq(a2)

        cap_loss = []
        im_loss = []

        for i in range(batch_size):
            cap_sims_ap_, cap_sims_an_ = [], []
            im_sims_ap_, im_sims_an_ = [], []
            cap_sims_ap_.append(scores[i][mask[i]])
            cap_sims_an_.append(scores[i][mask[i] == 0])
            im_sims_ap_.append(scoresT[i][mask[i]])
            im_sims_an_.append(scoresT[i][mask[i] == 0])

            cap_sims_ap_pair = cap_sims_ap_[0][cap_sims_ap_[0] < max(cap_sims_an_[0]) + self.Delta]
            cap_sims_an_pair = cap_sims_an_[0][cap_sims_an_[0] > min(cap_sims_ap_[0]) - self.Delta]
            im_sims_ap_pair = im_sims_ap_[0][im_sims_ap_[0] < max(im_sims_an_[0]) + self.Delta]
            im_sims_an_pair = im_sims_an_[0][im_sims_an_[0] > min(im_sims_ap_[0]) - self.Delta]

            cap_an_num = len(cap_sims_an_pair)
            cap_ap_num = len(cap_sims_ap_pair)
            im_an_num = len(im_sims_an_pair)
            im_ap_num = len(im_sims_ap_pair)

            if cap_an_num < 1 or cap_ap_num < 1 or im_an_num < 1 or im_ap_num < 1:
                continue

            # weighting step
            cap_pos_loss = 1.0 / self.gamma_1 * torch.log(
                torch.sum(torch.exp(self.gamma_2 - self.gamma_1 * cap_sims_ap_pair)))
            cap_neg_loss = 1.0 / self.gamma_1 * torch.log(
                torch.sum(torch.exp(self.gamma_1 * cap_sims_an_pair - self.gamma_2)))

            im_pos_loss = 1.0 / self.gamma_1 * torch.log(
                torch.sum(torch.exp(self.gamma_2 - self.gamma_1 * im_sims_ap_pair)))
            im_neg_loss = 1.0 / self.gamma_1 * torch.log(
                torch.sum(torch.exp(self.gamma_1 * im_sims_an_pair - self.gamma_2)))

            cap_loss.append(cap_pos_loss + cap_neg_loss)
            im_loss.append(im_pos_loss + im_neg_loss)

        if len(cap_loss) == 0 or len(im_loss) == 0:
            return torch.zeros([], requires_grad=True, dtype=torch.float64).cuda()
        cost_cap = sum(cap_loss) / batch_size
        cost_img = sum(im_loss) / batch_size
        loss = cost_cap + cost_img
        return loss

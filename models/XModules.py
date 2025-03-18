import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus



def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(mm, s):
    """Cosine similarity between all the motion and sentence pairs
    """
    return mm.mm(s.t())


def  js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output, dim=-1)
        q_output = F.softmax(q_output, dim=-1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GatedQueryAttLayer(nn.Module):
    def __init__(self, embed_size, h, is_share, drop=None):
        super(GatedQueryAttLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear] 
        else:
            self.linears = clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)

        self.fc_q = nn.Linear(self.d_k, self.d_k)
        self.fc_k = nn.Linear(self.d_k, self.d_k)
        self.fc_g = nn.Linear(self.d_k, self.d_k*2)

    def forward(self, inp, mask=None):
        nbatches = inp.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]
        # gate
        G = self.fc_q(query) * self.fc_k(key)
        M = torch.sigmoid(self.fc_g(G)) # (bs, h, num_region, d_k*2)
        query = query * M[:, :, :, :self.d_k]
        key = key * M[:, :, :, self.d_k:]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value) 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return x

class AGSA(nn.Module):
    ''' Adaptive Gating Self-Attention module '''
    def __init__(self, num_layers, embed_size, h=1, is_share=False, drop=None):
        super(AGSA, self).__init__()
        self.num_layers = num_layers
        self.bns = clones(nn.BatchNorm1d(embed_size), num_layers)
        self.dropout = clones(nn.Dropout(drop), num_layers)
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.att_layers = clones(GatedQueryAttLayer(embed_size, h, is_share, drop=drop), num_layers)

    def forward(self, rgn_emb, pos_emb=None, mask=None):
        ''' imb_emb -- (bs, num_r, dim), pos_emb -- (bs, num_r, num_r, dim) '''
        bs, num_r, emb_dim = rgn_emb.size()
        if pos_emb is None:
            x = rgn_emb
        else:
            x = rgn_emb * pos_emb
        
        # 1st layer
        x = self.att_layers[0](x, mask)    #(bs, r, d)
        x = (self.bns[0](x.view(bs*num_r, -1))).view(bs, num_r, -1) 
        agsa_emb = rgn_emb + self.dropout[0](x)

        # 2nd~num_layers
        for i in range(self.num_layers - 1):
            x = self.att_layers[i+1](agsa_emb, mask) #(bs, r, d)
            x = (self.bns[i+1](x.view(bs*num_r, -1))).view(bs, num_r, -1) 
            agsa_emb = agsa_emb + self.dropout[i+1](x)

        return agsa_emb


class SelfEncoder(nn.Module):
    def __init__(self, config, embed_size, head, drop=0.0):
        super(SelfEncoder, self).__init__()
        self.mapping = nn.Linear(config.hidden_size, embed_size)
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        # MLP
        self.fc1 = nn.Linear(embed_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, input):
        x = self.mapping(input)    #(bs, token_num, final_dim)
        bs, token_num = x.size()[:2]
        agsa_emb = self.agsa(x)
        x = self.fc2(self.dropout(F.relu(self.fc1(agsa_emb))))
        x = (self.bn(x.view(bs*token_num, -1))).view(bs, token_num, -1)
        x = agsa_emb + self.dropout(x)    # context-enhanced word embeddings

        return x


class SoftContrastiveLoss(nn.Module):
    """
    Compute triplet loss
    """
    def __init__(self, alpha, margin=0, max_violation=False, threshold_hetero=1.0, threshold_homo=1.0, **kwargs):
        super(SoftContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.alpha = alpha
        self.max_violation = max_violation
        self.threshold_hetero = threshold_hetero
        self.threshold_homo = threshold_homo

    def forward(self, motion_emb, text_emb):
        return self.compute(motion_emb, text_emb) + self.compute(text_emb, motion_emb)

    def compute(self, emb1, emb2):
        # compute motion-sentence score matrix
        scores = self.sim(emb1, emb2)    # (bsz, bsz)

        # Soft hard negative mining
        # MildTrip loss function implementation
        if self.max_violation:
            scores_emb1 = self.sim(emb1, emb1)    # (bsz, bsz)
            scores_emb2 = self.sim(emb2, emb2)    # (bsz, bsz)
            mask_emb1 = (scores_emb1 > self.threshold_hetero) & (
                scores_emb1 < 1 - 1e-6)
            mask_emb2 = (scores_emb2 > self.threshold_homo) & (
                scores_emb2 < 1 - 1e-6)
            scores[mask_emb1 | mask_emb2] = 0

        # positive-pair score
        diagonal = scores.diag().view(-1, 1)

        # Expand to the right
        d = diagonal.expand_as(scores)
        # Given emb1 retrieves the number of entries in emb2
        # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        cost_emb1 = (self.margin + scores - d).clamp(min=0)

        # clear positive pairs
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(emb1.device)
        cost_emb1 = cost_emb1.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            # always hardest negative
            cost_emb1 = cost_emb1.max(1)[0]

        return self.alpha * cost_emb1.sum()


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin=0.2, measure='cosine', max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.measure = measure
        self.max_violation = max_violation

    def forward(self, img_rep, txt_rep):
        """
            image_rep: (bs, 50, 768) -> attention weighted && reverse attention-> (bs, 4, 2, 768)
            label_rep: (bs, 4, 768) -> (bs, 4, 1, 768)
            where dim = -2 can be regarded as batch size
        """
        if self.measure == 'cosine':
            # shape: (bs, 4, 2)
            # CCR Part
            scores = self.cosine_sim_v1(img_rep, txt_rep).squeeze()
            # scores[0] representation positive result
            cost_ccr = (self.margin + scores - scores[:, :, 0].unsqueeze(-1)).clamp(0)

            # CCR mask
            mask = torch.tensor([1., 0.]).unsqueeze(0).unsqueeze(1).expand_as(scores) == 1.
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_ccr = cost_ccr.masked_fill_(I, 0)

            # shape: (bs, 4, 4)
            # CCS Part
            scores = self.cosine_sim_v2(img_rep, txt_rep)
            diagonal = torch.diagonal(scores, dim1=-2, dim2=-1).view(scores.size(0), -1, 1)
            d = diagonal.expand_as(scores)
            cost_ccs = (self.margin + scores - d).clamp(min=0)

            # CCS mask
            mask = torch.eye(scores.size(-1)).expand_as(scores) > .5
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_ccs = cost_ccs.masked_fill_(I, 0)

            if self.max_violation:
                cost_ccs = cost_ccs.max(-1)[0]
            return self.alpha * cost_ccr.sum() + self.beta * cost_ccs.sum()

    @staticmethod
    def cosine_sim_v1(img_rep, txt_rep):
        return torch.matmul(img_rep, txt_rep.transpose(-1, -2).contiguous()) / math.sqrt(img_rep.size(-1))

    @staticmethod
    def cosine_sim_v2(img_rep, txt_rep):
        img_rep = img_rep[:, :, 0, :]
        txt_rep = txt_rep.squeeze()
        return torch.matmul(img_rep, txt_rep.transpose(-1, -2).contiguous()) / math.sqrt(img_rep.size(-1))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):    # Squeeze and Excitation
        a, b, _ = x.size()    # x:(bsz, 3, 768)
        y = self.avg_pool(x).view(a, b)    # (bsz, 3)
        y = self.fc(y).view(a, b, 1)    # (bsz, 3, 1)
        return x * y.expand_as(x), y.squeeze(-1)



class CrossModalAlignment(nn.Module):
    def __init__(self, config, args):
        super(CrossModalAlignment, self).__init__()
        self.config = config
        self.args = args
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.closs = ContrastiveLoss(alpha=args.alpha, beta=0, margin=args.margin)


    def forward(self, text_emb, image_emb):
        """
        inputs :
            text_emb : input feature maps( B X 128 X 768 )
            image_emb : input feature maps( B X 50 X 768 )
        returns :
            out : ( B X 128 X 768 )
        """
        query_layer = self.query(text_emb)       # (bsz, 128, 768)
        key_layer = self.key(image_emb)          # (bsz, 50, 768)
        value_layer = self.value(image_emb)      # (bsz, 50, 768)

        # (bsz, 128, 50)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))
        attn_score = attention_scores / math.sqrt(self.config.hidden_size)

        # Softmax attention   (bsz, 128, 768)
        attn_score = torch.softmax(100 * attn_score, dim=-1)
        text_img_rep_init = torch.bmm(attn_score, value_layer)

        # reverse Softmax attention    (bsz, 128, 768)
        reverse_score = torch.softmax(100 * (1 - attn_score), dim=-1)
        reverse_text_img_rep_init = torch.bmm(reverse_score, value_layer)

        # text_img_rep_init, reverse_text_img_rep_init = SCAN_attention(text_emb, image_emb, smooth=9.0)
        # # (32, 128, 768)
        # text_img_rep_init, reverse_text_img_rep_init = self.text_img_CrossAttention(text_emb, image_emb, image_attention_mask)
        # (32, 128, 1, 768)
        text_img_rep = self.fc_1(text_img_rep_init).unsqueeze(-2)
        reverse_text_img_rep = self.fc_2(reverse_text_img_rep_init).unsqueeze(-2)
        # (bsz, 128, 2, 768)
        total_text_img_rep = torch.cat((text_img_rep, reverse_text_img_rep), dim=-2)

        text_img_loss = self.closs(torch.nn.functional.normalize(total_text_img_rep),
                             torch.nn.functional.normalize(text_emb.unsqueeze(-2)))

        return text_img_rep_init, text_img_loss


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """
    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(True),
            nn.Linear(768, z_dim * 2),
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)

class AmbiguityLearning(nn.Module):
    def __init__(self, weight_js=0.5):
        super(AmbiguityLearning, self).__init__()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()
        self.weight_js = weight_js

    def forward(self, text_encoding, image_encoding, weight_input):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = torch.sigmoid(skl)

        # 拼接三个通道的模糊度作为target   (1-skl, skl, 1-skl)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        weight_target = torch.cat((weight_uni, weight_corre, weight_uni), dim=1)    # (bsz, 3)

        # JS散度 -> 双向KL散度    (bsz, 3)
        js_loss = - js_div(weight_input, weight_target)
        # js_loss = js_div(weight_input, weight_target)

        # KLD = nn.KLDivLoss(reduction="batchmean")
        # input = F.log_softmax(weight_input, dim=-1)
        # target = F.softmax(weight_target, dim=-1)
        # js_loss = KLD(input, target)

        # # input should be a distribution in the log space
        # input = F.log_softmax(target, dim=0)
        # # Sample a batch of distributions. Usually this would come from the dataset
        # target = F.softmax(skl, dim=0)
        # output = kl_loss(input, target)
        return self.weight_js * js_loss



def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks    # split_size:80
    sizes_list = [split_size] * chunks      # list:20 [80, 80, ..., 80]
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim) # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1]<0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j-1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list

def get_chunks(x,sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1,begin,s)    # (32, 80)
        out.append(y)     # list:20
        begin += s
    return out


class Block(nn.Module):
    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            chunks=20,
            rank=15,
            shared=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.,
            pos_norm='before_cat'):
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size*rank)     # (80, 1200)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size*rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])   # (32, 1600)
        x1 = self.linear1(x[1])   # (32, 1600)
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)    # list:20  (32, 80)
        x1_chunks = get_chunks(x1, self.sizes_list)    # list:20  (32, 80)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1):
            x0_c = x0_chunks[chunk_id]    # (32, 80)
            x1_c = x1_chunks[chunk_id]    # (32, 80)
            m = m0(x0_c) * m1(x1_c)   # bsize x split_size*rank   (32, 80 * 15)
            m = m.view(bsize, self.rank, -1)    # (32, 15, 80)
            z = torch.sum(m, 1)     # (32, 80)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))    # (32, 80)
                z = F.normalize(z,p=2)     # (32, 80)
            zs.append(z)     # list:20
        z = torch.cat(zs,1)    # (32, 1600)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)    # (32, 768)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class DiffLoss(nn.Module):

    def __init__(self, args):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        # input1 (B,N,D)    input2 (B,N,D)

        batch_size = input1.size(0)
        N = input1.size(1)
        input1 = input1.contiguous().view(batch_size, -1)  # (B,N*D)
        input2 = input2.contiguous().view(batch_size, -1)  # (B, N*D)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)  # (1,N*D)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)  # (1,N*D)
        input1 = input1 - input1_mean  # (B,N*D)
        input2 = input2 - input2_mean  # (B,N*D)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()  # (B,1)
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)  # (B,N*D)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()  # (B,1)
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)  # (B,N*D)

        diff_loss = 1.0 / (torch.mean(torch.norm(input1_l2 - input2_l2, p=2, dim=1)))

        return diff_loss

